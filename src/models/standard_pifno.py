"""Module for the standard PIFNO solver without adaptive weighting."""

from itertools import chain
from typing import Any, Dict, Mapping, Tuple, cast

import torch

from .base import BaseModule
from .components.loss.burgers_loss import inviscid_burgers as burgers
from .components.loss.utils import LpLoss


class StandardPIFNOModule(BaseModule):
    """Solver class for the 1D Burgers' equation using a neural network.

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
        loss_method=burgers,
        compile_model: bool = True,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        lambda_ic: float = 5,
        lambda_f: float = 1,
        lambda_data: float = 5,
        padding: int = 5,
        spatial_padding: int = 0,
    ):
        """Initialize the StandardPIFNOModule."""
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net
        self.loss_method = loss_method
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
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        is_training: bool = False,
    ):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        input_batch, output = batch
        prediction = self._normalize_and_pad_input(input_batch)
        if output.ndim == 4 or output.ndim == 5:
            output = output.squeeze(-1)
        if prediction.ndim == 4 or prediction.ndim == 5:
            prediction = prediction.squeeze(-1)

        data_loss = LpLoss()(prediction, output)

        initial_condition_gt = output[:, 0, :]  # u(x, t=0) ground truth
        initial_condition_pred = prediction[:, 0, :]  # u(x, t=0) predicted

        ic_loss = LpLoss()(initial_condition_pred, initial_condition_gt)

        physics_loss = self.loss_method(
            u=prediction,
            input_batch=input_batch,
            output=output,
            current_epoch=self.current_epoch,
            dt=self.dt,
            dx=self.dx,
            dy=self.dy,
        )

        total_loss = (
            self.hparams.lambda_ic * ic_loss
            + self.hparams.lambda_f * physics_loss
            + self.hparams.lambda_data * data_loss
        )
        return (
            total_loss,
            data_loss,
            physics_loss,
            ic_loss,
        )

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ):
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        (
            total_loss,
            data_loss,
            physical_loss,
            ic_loss,
        ) = self.model_step(batch=batch, is_training=True)

        self.log_dict(
            {
                "train/loss": total_loss,
                "train/ic_loss": ic_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log_dict(
            cast(
                Mapping[str, Any],
                {
                    "train/data_loss": data_loss,
                    "train/physical_loss": physical_loss,
                },
            ),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return total_loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        (total_loss, data_loss, physical_loss, ic_loss) = self.model_step(
            batch=batch
        )

        self.log_dict(
            {
                "val/loss": total_loss,
                "val/ic_loss": ic_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log_dict(
            cast(
                Mapping[str, Any],
                {
                    "val/data_loss": data_loss,
                    "val/physical_loss": physical_loss,
                },
            ),
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
            scheduler_instance = scheduler(optimizer=optimizer)
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
