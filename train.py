from typing import Any

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

import wandb
from app.conf_resolver import register_resolvers
from app.data import SalChartQA

register_resolvers()

torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    data: SalChartQA = instantiate(cfg.data)
    model: L.LightningModule = instantiate(cfg.model)
    trainer: L.Trainer = instantiate(cfg.trainer)

    if trainer.logger:
        hparams: dict[str, Any] = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )  # type: ignore
        hparams.update(
            {
                "split": {
                    "train": data.train_components,
                    "val": data.val_components,
                }
            }
        )
        trainer.logger.log_hyperparams(hparams)

    trainer.fit(model, data)
    trainer.test(model, data, ckpt_path="best")

    if isinstance(trainer.logger, WandbLogger):
        wandb.finish()


if __name__ == "__main__":
    train()
