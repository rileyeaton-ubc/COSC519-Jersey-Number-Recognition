#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from pathlib import Path

from omegaconf import DictConfig, open_dict
import hydra
from hydra.core.hydra_config import HydraConfig

from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import summarize

from strhub.data.module import SceneTextDataModule
from strhub.models.base import BaseSystem
from strhub.models.utils import get_pretrained_weights

import math
import matplotlib.pyplot as plt
import numpy as np

k = 20

class LossCallback(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_losses = []
        self.single_epoch_losses = []
        self.epoch_stages = []
        self.moving_average_losses = []
        self.figure_num = 0
        self.moving_avg_loss = 0

    def moving_average(self, a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def on_after_backward(self, trainer, pl_module):
        # Ensure the model has computed the loss
        if pl_module.trainer.logged_metrics.get('loss', None) is not None:
            # Convert the GPU tensor to a Python float
            loss_value = pl_module.trainer.logged_metrics['loss'].detach().cpu().item()
            self.epoch_stages.append(trainer.current_epoch)
            self.single_epoch_losses.append(loss_value)
            self.epoch_losses.append(loss_value)
            if len(self.epoch_losses) < k:
                return
            self.moving_avg_loss = np.mean(self.epoch_losses[-k:])
            self.moving_average_losses.append(self.moving_avg_loss)
            plt.plot(self.moving_average_losses, color="deepskyblue")
            # plt.plot(self.epoch_losses, color="orange")
            plt.xlabel("Batch") 
            plt.ylabel("Loss")
            plt.title(f'PARSeq Fine-Tuning Training Loss, 20 Batch Moving Average [Epoch {trainer.current_epoch}]')
            plt.savefig(f'loss_data/images/loss_epoch{trainer.current_epoch}_{self.figure_num}.png')
            self.figure_num += 1

    def on_train_epoch_end(self, trainer, pl_module, dataloader_idx=None):
        epoch = trainer.current_epoch
        self.figure_num = 0

        file_path = f'loss_data/losses_epoch_{epoch}.txt'
        with open(file_path, 'w') as f:
            f.write("[")
            for loss_value in self.single_epoch_losses:
                f.write(f"{loss_value},")
            f.write("]")
        # Optionally, log the file path or summary info
        print(f"Saved after-backward losses for epoch {epoch} to {file_path}")
        # Reset losses for the next epoch
        self.single_epoch_losses = []

    # def on_validation_end(self, trainer, pl_module):
    #     if 'val_loss' in trainer.callback_metrics:
    #         val_loss_value = trainer.callback_metrics['val_loss'].detach().cpu().item()
    #         self.val_losses.append(val_loss_value)

# Copied from OneCycleLR
def _annealing_cos(start, end, pct):
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out


def get_swa_lr_factor(warmup_pct, swa_epoch_start, div_factor=25, final_div_factor=1e4) -> float:
    """Get the SWA LR factor for the given `swa_epoch_start`. Assumes OneCycleLR Scheduler."""
    total_steps = 1000  # Can be anything. We use 1000 for convenience.
    start_step = int(total_steps * warmup_pct) - 1
    end_step = total_steps - 1
    step_num = int(total_steps * swa_epoch_start) - 1
    pct = (step_num - start_step) / (end_step - start_step)
    return _annealing_cos(1, 1 / (div_factor * final_div_factor), pct)

@hydra.main(config_path='configs', config_name='main', version_base='1.2')
def main(config: DictConfig):
    trainer_strategy = None
    with open_dict(config):
        # Resolve absolute path to data.root_dir
        config.data.root_dir = hydra.utils.to_absolute_path(config.data.root_dir)
        # Special handling for GPU-affected config
        gpu = config.trainer.get('accelerator') == 'gpu'
        devices = config.trainer.get('devices', 0)
        if gpu:
            # Use mixed-precision training
            config.trainer.precision = 16
        if gpu and devices > 1:
            # Use DDP
            config.trainer.strategy = 'ddp'
            # DDP optimizations
            trainer_strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
            # Scale steps-based config
            config.trainer.val_check_interval //= devices
            if config.trainer.get('max_steps', -1) > 0:
                config.trainer.max_steps //= devices

    # Special handling for PARseq
    if config.model.get('perm_mirrored', False):
        assert config.model.perm_num % 2 == 0, 'perm_num should be even if perm_mirrored = True'

    model: BaseSystem = hydra.utils.instantiate(config.model)
    # If specified, use pretrained weights to initialize the model
    if config.pretrained is not None:
        model.load_state_dict(get_pretrained_weights(config.pretrained))
    print(summarize(model, max_depth=1 if model.hparams.name.startswith('parseq') else 2))

    datamodule: SceneTextDataModule = hydra.utils.instantiate(config.data)

    checkpoint = ModelCheckpoint(monitor='val_accuracy', mode='max', save_top_k=3, save_last=True,
                                 filename='{epoch}-{step}-{val_accuracy:.4f}-{val_NED:.4f}')
    
    loss_callback = LossCallback()

    swa_epoch_start = 0.75
    swa_lr = config.model.lr * get_swa_lr_factor(config.model.warmup_pct, swa_epoch_start)
    swa = StochasticWeightAveraging(swa_lr, swa_epoch_start)
    cwd = HydraConfig.get().runtime.output_dir if config.ckpt_path is None else \
        str(Path(config.ckpt_path).parents[1].absolute())
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=TensorBoardLogger(cwd, '', '.'),
                                               strategy=trainer_strategy, enable_model_summary=False,
                                               callbacks=[checkpoint, swa, loss_callback])
    trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)


if __name__ == '__main__':
    main()
