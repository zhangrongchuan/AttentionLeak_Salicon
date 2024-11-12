from copy import deepcopy
from typing import Literal

import lightning as L
import torch
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import nn
from torch.optim import lr_scheduler, optimizer
from torchvision.transforms.v2.functional import normalize, to_pil_image

import wandb.plot


class UncertaintyWeightingMultiTask(nn.Module):
    def __init__(self, num_tasks: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks), requires_grad=True)

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.log_vars) @ losses + self.log_vars.sum()


class LinearWeightingMultiTask(nn.Module):
    def __init__(self, num_tasks: int):
        super().__init__()

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        return losses.sum()


T_TASKS_LOSS_AGGREGATIONS = Literal["uncertainty_weighting", "linear"]

TASKS_LOSS_AGGREGATIONS: dict[T_TASKS_LOSS_AGGREGATIONS, type[nn.Module]] = {
    "uncertainty_weighting": UncertaintyWeightingMultiTask,
    "linear": LinearWeightingMultiTask,
}


class MultiTaskModel(L.LightningModule):
    def __init__(
        self,
        optimizer: type[optimizer.Optimizer],
        backbone: nn.Module,
        loss: nn.Module,
        labels: list[str],
        num_classes: list[int],
        class_names: list[list[str]],
        loss_aggregation: T_TASKS_LOSS_AGGREGATIONS,
        lr_scheduler: type[lr_scheduler.LRScheduler] | None = None,
        train_metrics: dict[str, type[torchmetrics.Metric]] | None = None,
        test_metrics: dict[str, type[torchmetrics.Metric]] | None = None,
    ):
        super().__init__()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss = loss
        self.labels = labels
        self.num_classes = num_classes
        self.class_names = class_names

        train_metrics = train_metrics or {}
        test_metrics = test_metrics or {}
        self.metrics = nn.ModuleDict(
            {
                f"train/{label}_metrics": nn.ModuleDict(
                    deepcopy(
                        {
                            name: metric(num_classes=self.num_classes[i])
                            for name, metric in train_metrics.items()
                        }
                    )
                )
                for i, label in enumerate(self.labels)
            }
            | {
                f"{step}/{label}_metrics": nn.ModuleDict(
                    deepcopy(
                        {
                            name: metric(num_classes=self.num_classes[i])
                            for name, metric in test_metrics.items()
                        }
                    )
                )
                for step in ["val", "test"]
                for i, label in enumerate(self.labels)
            }
        )

        assert hasattr(backbone, "num_features")
        self.backbone = backbone
        self.input_mean = self.backbone.pretrained_cfg["mean"]
        self.input_std = self.backbone.pretrained_cfg["std"]
        self.task_heads = nn.ModuleDict(
            {
                label: nn.Linear(backbone.num_features, n)
                for label, n in zip(labels, num_classes)
            }
        )
        self.loss_agg = TASKS_LOSS_AGGREGATIONS[loss_aggregation](len(labels))

        self.table = wandb.Table(
            columns=["image_name", "saliency_map", "question"]
            + [f"{label}_{x}" for label in self.labels for x in ["gt", "pred"]]
        )

    def forward(self, inp: torch.Tensor) -> dict[str, torch.Tensor]:
        inp = normalize(inp, mean=self.input_mean, std=self.input_std)
        features = self.backbone(inp)
        task_logits = {label: head(features) for label, head in self.task_heads.items()}
        return task_logits

    def step(self, batch, batch_idx, step: str) -> dict[str, torch.Tensor]:
        task_logits = self(batch["saliency_map"])
        losses = []
        for label, logits in task_logits.items():
            loss = self.loss(logits, batch[label])
            self.log(f"{step}/{label}/loss", loss)
            self.log_metrics(logits, batch[label], f"{step}/{label}")
            losses.append(loss)

        total_loss = self.loss_agg(torch.stack(losses))
        self.log(f"{step}/total_loss", total_loss)
        return {"loss": total_loss} | task_logits

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        results = self.step(batch, batch_idx, "test")
        self.log_wandb_table(batch, results)
        return results

    def on_test_epoch_end(self) -> None:
        # TODO log table
        if not isinstance(self.logger, WandbLogger):
            return
        self.logger.experiment.log({"test/predictions": self.table})

    def log_wandb_table(
        self, batch: dict[str, torch.Tensor], results: dict[str, torch.Tensor]
    ):
        for i in range(batch["saliency_map"].shape[0]):
            image_name = batch["image_name"][i]
            saliency_map = wandb.Image(to_pil_image(batch["saliency_map"][i]))
            question = batch["question"][i]
            row = [image_name, saliency_map, question]
            for j, label in enumerate(self.labels):
                row.append(self.class_names[j][int(batch[label][i].item())])
                row.append(self.class_names[j][int(results[label][i].argmax().item())])
            self.table.add_data(*row)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())  # type: ignore
        if self.lr_scheduler is None:
            return optimizer
        scheduler = self.lr_scheduler(optimizer)
        return OptimizerLRSchedulerConfig(
            {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/total_loss",
                },
            }
        )

    def log_metrics(self, logits, labels, step):
        for metric_name, metric in self.metrics[f"{step}_metrics"].items():
            assert isinstance(metric, torchmetrics.Metric)
            metric_name = f"{step}/{metric_name}"
            if self.trainer.global_step == 0 and isinstance(self.logger, WandbLogger):
                self.logger.experiment.define_metric(
                    metric_name,
                    summary="max" if metric.higher_is_better else "min",
                )
            metric(logits, labels)
            self.log(
                metric_name,
                metric,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
