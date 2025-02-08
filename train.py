from typing import Any

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

import wandb
from app.conf_resolver import register_resolvers
from app.data import SaliconDataset
import os
import sys

import pdb

register_resolvers()

torch.set_float32_matmul_precision("high")

# 将工作目录设置为脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    # pdb.set_trace()
    data: SalChartQA = instantiate(cfg.data)
    model: L.LightningModule = instantiate(cfg.model)
    trainer: L.Trainer = instantiate(cfg.trainer,log_every_n_steps=1)

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
        
    # pdb.set_trace()
    trainer.fit(model, data)
    # trainer.test(model, data, ckpt_path="best")
    trainer.test(model, data)

    if isinstance(trainer.logger, WandbLogger):
        wandb.finish()


if __name__ == "__main__":
    # cfg = OmegaConf.load("conf/config.yaml")
    train()