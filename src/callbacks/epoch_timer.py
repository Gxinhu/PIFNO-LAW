import time

import torch
from lightning.pytorch import Callback


class EpochTimingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_epoch_start_time: float = 0
        self.validation_epoch_start_time: float = 0
        self.test_batch_start_time: float = 0

    def _get_monotonic_time(self, trainer):
        if trainer.strategy.accelerator.is_available:
            torch.cuda.synchronize()
        return time.monotonic()

    # --- Training Time Measurement ---
    def on_train_epoch_start(self, trainer, pl_module):
        """Called at the start of a training epoch."""
        self.train_epoch_start_time = self._get_monotonic_time(trainer)

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of a training epoch."""
        end_time = self._get_monotonic_time(trainer)
        elapsed_secs = end_time - self.train_epoch_start_time

        pl_module.log("epoch_metrics/train_epoch_time_sec", elapsed_secs)

    def on_validation_epoch_start(self, trainer, pl_module):
        """Called at the start of a validation epoch."""
        self.validation_epoch_start_time = self._get_monotonic_time(trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of a validation epoch."""
        end_time = self._get_monotonic_time(trainer)
        elapsed_secs = end_time - self.validation_epoch_start_time

        pl_module.log("epoch_metrics/validation_epoch_time_sec", elapsed_secs)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Called just before the test batch is processed."""
        self.test_batch_start = self._get_monotonic_time(trainer)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called just after the test batch is processed."""
        elapsed_ms = (
            self._get_monotonic_time(trainer) - self.test_batch_start
        ) * 1000
        pl_module.log(
            "batch_metrics/test_batch_time_ms", elapsed_ms, on_step=True
        )
