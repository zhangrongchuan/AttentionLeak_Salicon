from copy import deepcopy
from typing import Literal

import pdb
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
import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()

# Replace loss at model initialization



class MultiTaskModel(L.LightningModule):
    def __init__(
        self,
        optimizer: type[optimizer.Optimizer],
        backbone: nn.Module,
        # loss: nn.Module,
        loss: FocalLoss,
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
        self.met = [train_metrics,test_metrics]
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
        for param in self.backbone.parameters():
            param.requires_grad = False  # 冻结所有层
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True  # 只训练最后一层
        # for param in self.backbone.parameters():
        #     print(param.requires_grad)
        self.input_mean = self.backbone.pretrained_cfg["mean"]
        self.input_std = self.backbone.pretrained_cfg["std"]
        self.task_heads = nn.ModuleDict(
            {
                label: nn.Sequential(
                        nn.Dropout(p=0.5),  # 50% probability of randomly discarding neurons
                        nn.Linear(backbone.num_features, n)
                    )
                # label: nn.Linear(backbone.num_features, n)
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
        # print(features,features.shape)
        task_logits = {label: head(features) for label, head in self.task_heads.items()}
        # print([(label, head(features)) for label, head in self.task_heads.items()])
        # print("--------------------------------------------------------------------")
        return task_logits

    def step(self, batch, batch_idx, step: str) -> dict[str, torch.Tensor]:
        task_logits = self(batch["saliency_map"])
        # print(self.num_classes)
        # print(task_logits)
        losses = []
        for label, logits in task_logits.items():#Logits are predictions without softmax.
            loss = self.loss(logits, batch[label])
            # print(logits,batch[label],label, batch)
            self.log(f"{step}/{label}/loss", loss)
            self.log_metrics(logits, batch[label], f"{step}/{label}")
            losses.append(loss)
        # print("--------------------------------------------------------------------")

        total_loss = self.loss_agg(torch.stack(losses))
        self.log(f"{step}/total_loss", total_loss)
        return {"loss": total_loss} | task_logits

    def training_step(self, batch, batch_idx):
        # print(f"Batch keys: {batch.keys()}")  # Print what keys are in the batch
        # print(f"Saliency map shape: {batch['saliency_map'].shape}")  # Confirmation of input data
        # for key, value in batch.items():
        #     print(f"{key}: {value.shape}")  # Check data for all keys
        res=self.step(batch, batch_idx, "train")
        # print(res)
        return res
        # return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        res=self.step(batch, batch_idx, "val")
        # results = self.step(batch, batch_idx, "test")
        # self.log_wandb_table(batch, res)
        # print(res)

    def test_step(self, batch, batch_idx):
        results = self.step(batch, batch_idx, "test")
        # print(results)
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
            # acc=torchmetrics.Accuracy(task="multiclass", num_classes=12,average=None).to("cuda")
            # acc_class=acc(logits, labels)
            metric(logits, labels)
            self.log(
                metric_name,
                metric,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
