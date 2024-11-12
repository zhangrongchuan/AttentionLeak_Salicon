import json
from functools import partial
from pathlib import Path
from typing import Literal

import lightning as L
import networkx as nx
import polars as pl
import torch
import typer
from datasets import Dataset, DatasetDict
from PIL import Image
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from torchvision import tv_tensors
from torchvision.transforms import v2

app = typer.Typer()


def get_components(dataframe: pl.DataFrame) -> dict[str, int]:
    """Return a mapping from image name to connected component id.""

    The dataset consists of saliency maps from users on specific plots.
    Each participant was shown several plots and asked a question about it.
    Each component is a cluster of participants who answered questions about the same plots.
    The component can be used to split the dataset into train and validation sets,
    without leaking information about participants and plots between them.

    Args:
        dataframe: A DataFrame with columns "participant_id" and "image_name".

    Returns:
        A dictionary mapping image names to component ids.
    """
    graph = nx.Graph()
    for participant_id, img_name in dataframe[
        ["participant_id", "image_name"]
    ].iter_rows():
        graph.add_edge(participant_id, img_name)
    components = nx.connected_components(graph)
    components = list(
        filter(lambda x: x.endswith(".png"), component) for component in components
    )
    print(f"Found {len(components)} components.")
    image_to_component = {
        img_name: component_id
        for component_id, component in enumerate(components)
        for img_name in component
    }
    return image_to_component


def prepare_dataframe(dataframe: pl.DataFrame, questions) -> pl.DataFrame:
    name_to_component = get_components(dataframe)
    dataframe = dataframe.with_columns(
        pl.col("image_name")
        .replace(name_to_component)
        .cast(pl.UInt8)
        .alias("component")
    )
    new_df = pl.from_dicts(
        [
            {"image_name": x, "question_number": y, "question": z}
            for x, yz in questions.items()
            for y, z in yz.items()
        ]
    )
    dataframe = (
        dataframe.group_by(
            [
                "image_name",  # used to join with question number
                "image_type",  # potential attribute to infer
                "question",  # used to join with question number
                "component",  # used to split without leaking information of plots or participatns
                "is_chart_simple",  # potential attribute to infer
                "question_type",  # potential attribute to infer
            ]
        )
        .agg(pl.col("number_of_clicks").median())  # potential attribute to infer
        .with_columns(
            pl.col("number_of_clicks")
            .qcut(
                [0.25, 0.75], labels=["easy", "medium", "hard"]
            )  # regression -> classification
            .over("image_type")
            .alias("complexity")
        )
        .join(new_df, on=["image_name", "question"], how="inner")
    )
    return dataframe


@app.command()
def prepare_data(input: Path, output: Path, correct_only: bool = False):
    input = input.expanduser()
    output = output.expanduser()

    dataframe = pl.read_csv(input / "unified_approved.csv")
    questions = json.loads((input / "image_questions.json").read_text())
    dataframe = prepare_dataframe(dataframe, questions)
    dataframe = dataframe.with_columns(
        pl.col("question_type").replace(["Other", "CL", "CP", "U"], None)
    )

    def get_image_path(row):
        if correct_only:
            return dict(
                image_path=str(
                    input
                    / "saliency_ans"
                    / "heatmaps"
                    / f"{row["image_name"].split(".")[0]}_{row["question_number"]}_True.png"
                )
            )
        else:
            return dict(
                image_path=str(
                    input
                    / "saliency_all"
                    / "heatmaps"
                    / f"{row["image_name"].split(".")[0]}_{row["question_number"]}.png"
                )
            )

    def load_image(path: str):
        return dict(saliency_map=Image.open(path))

    dataset = (
        Dataset.from_polars(dataframe)
        .class_encode_column("image_type")
        .class_encode_column("is_chart_simple")
        .class_encode_column("question_type")
        .class_encode_column("complexity")
        .map(get_image_path, desc="Get image paths")
        .filter(
            lambda x: Path(x).exists(),
            input_columns="image_path",
            desc="Filter missing images",
        )
        .map(load_image, input_columns="image_path", desc="Load images")
    )
    dataset_dict = DatasetDict(
        {
            "train": dataset.filter(lambda x: x < 120, input_columns="component"),
            "test": dataset.filter(lambda x: x >= 120, input_columns="component"),
        }
    )
    dataset_dict.save_to_disk(output)


def transform_wrapper(input: dict, transforms) -> dict:
    # transpose dict[str, list[x]] -> list[dict[str, x]]
    output = [dict(zip(input, col)) for col in zip(*input.values())]
    output = list(map(transforms, output))
    # transpose list[dict[str, x]] -> dict[str, list[x]]
    output = {k: [x[k] for x in output] for k in input.keys()}
    return output


SAMPLING_STRATEGIES = Literal["random", "balanced"]


def create_sampler(labels: torch.Tensor, strategy: SAMPLING_STRATEGIES) -> Sampler:
    match strategy:
        case "balanced":
            _, counts = torch.unique(labels, return_counts=True)
            weights = 1.0 / counts[labels]
            return WeightedRandomSampler(
                weights=weights, num_samples=len(labels), replacement=True
            )
        case _:
            raise ValueError(f"Unknown strategy {strategy}")


class SalChartQA(L.LightningDataModule):
    def __init__(
        self,
        path: Path,
        label: list[str] | str,
        dataloader: type[DataLoader[dict[str, torch.Tensor]]],
        train_transforms: v2.Compose,
        test_transforms: v2.Compose,
        sampling_strategy: SAMPLING_STRATEGIES,
        **kwargs,
    ) -> None:
        super().__init__()
        self.path = path
        self.labels = [label] if isinstance(label, str) else label
        self.dataloader_factory = dataloader
        self.sampling_strategy = sampling_strategy
        self.train_transforms = v2.Compose(
            [
                train_transforms,
                v2.ToDtype(
                    {
                        tv_tensors.Image: torch.float32,
                        "others": torch.long,
                    },
                    scale=True,
                ),
            ]
        )
        self.test_transforms = v2.Compose(
            [
                test_transforms,
                v2.ToDtype(
                    {
                        tv_tensors.Image: torch.float32,
                        "others": torch.long,
                    },
                    scale=True,
                ),
            ]
        )
        components = torch.randperm(120)
        self.train_components = set(components[:90].tolist())  # 60% of the data
        self.val_components = set(components[90:120].tolist())  # 20% of the data

    def prepare_data(self):
        # TODO: Download dataset and execute prepare_data command
        pass

    def setup(self, stage):
        if hasattr(self, "dataset"):
            return
        dataset = DatasetDict.load_from_disk(str(self.path)).filter(
            lambda *xs: all(x is not None for x in xs), input_columns=self.labels
        )  # filter excluded labels
        self.dataset = DatasetDict(
            train=dataset["train"].filter(
                lambda x: x in self.train_components,
                input_columns="component",
            ),
            val=dataset["train"].filter(
                lambda x: x in self.val_components,
                input_columns="component",
            ),
            test=dataset["test"],
        )

    def train_dataloader(self):
        dataset = self.dataset["train"].with_transform(
            partial(transform_wrapper, transforms=self.train_transforms),
            columns=["saliency_map"] + self.labels,
        )
        if self.sampling_strategy == "random":
            return self.dataloader_factory(dataset)  # type: ignore

        labels = torch.tensor(self.dataset["train"][self.labels[0]])
        sampler = create_sampler(labels, self.sampling_strategy)
        return self.dataloader_factory(dataset, sampler=sampler, shuffle=None)  # type: ignore

    def val_dataloader(self):
        dataset = self.dataset["val"].with_transform(
            partial(transform_wrapper, transforms=self.test_transforms),
            columns=["saliency_map"] + self.labels,
        )
        return self.dataloader_factory(dataset, shuffle=False)  # type: ignore

    def test_dataloader(self):
        dataset = self.dataset["test"].with_transform(
            partial(transform_wrapper, transforms=self.test_transforms),
            columns=["saliency_map", "question", "image_name"] + self.labels,
        )
        return self.dataloader_factory(dataset, shuffle=False)  # type: ignore


if __name__ == "__main__":
    app()
