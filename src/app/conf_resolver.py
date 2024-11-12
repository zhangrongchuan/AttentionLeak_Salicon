from datasets import ClassLabel, DatasetDict
from omegaconf import OmegaConf


def num_classes(path: str, label: str) -> int:
    dataset = DatasetDict.load_from_disk(str(path))["train"]
    assert isinstance(dataset.features[label], ClassLabel)
    return dataset.features[label].num_classes


def class_names(path: str, labels: str) -> list[str]:
    dataset = DatasetDict.load_from_disk(str(path))["train"]
    assert isinstance(dataset.features[labels], ClassLabel)
    return dataset.features[labels].names


def register_resolvers():
    OmegaConf.register_new_resolver("num_classes", num_classes)
    OmegaConf.register_new_resolver("class_names", class_names)
