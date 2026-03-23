"""Module for the base Fourier Neural Operator model."""

from itertools import chain
from typing import Any, Dict, Tuple, cast

import torch

from .base import BaseModule
from .components.loss.utils import LpLoss


class FNOModule(BaseModule):
    """Encapsulate the entire workflow for training and evaluating an FNO.

    This class encapsulates the entire workflow for training and evaluating
    a neural network model to solve the 1D Burgers' equation. It handles
    data loading, model initialization, training, validation, evaluation,
    and visualization.
    """

    net: torch.nn.Module

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        compile_model: bool = True,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        padding: int = 5,
        spatial_padding: int = 0,
    ):
        """Initialize the FNO module."""
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net
        self.padding = padding

    def configure_model(self):
        """Configure the model by recursively compiling it if requested."""
        if not self.hparams.compile_model:
            return

        if isinstance(self.net, torch._dynamo.eval_frame.OptimizedModule):
            return

        self.net = cast(
            torch.nn.Module,
            torch.compile(
                self.net,
                options={"shape_padding": True, "triton.cudagraphs": True},
            ),
        )

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the network."""
        return self.net(input_batch)

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tensor of losses.
        """
        # Pad input for spectral layers
        input_batch, output_batch = batch
        prediction = self._normalize_and_pad_input(input_batch)
        data_loss = LpLoss()(prediction, output_batch)
        return data_loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        data_loss = self.model_step(
            batch=batch,
        )

        self.log(
            "train/data_loss",
            data_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return data_loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        data_loss = self.model_step(
            batch=batch,
        )
        self.log(
            "val/data_loss",
            data_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.

        """
        optimizer = self.hparams.optimizer(params=chain(self.net.parameters()))
        scheduler = self.hparams.scheduler
        if scheduler is not None:
            scheduler_instance = scheduler(
                optimizer=optimizer,
                steps_per_epoch=self.trainer.estimated_stepping_batches,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler_instance,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return optimizer
