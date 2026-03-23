"""Module implementing the PIFNO-LAW network architectures."""

from itertools import chain
from typing import Any, Dict, Tuple, cast

import torch

from .base import BaseModule
from .components.loss.utils import LpLoss


class PIFNOLAWModule(BaseModule):
    """Physics-Informed Neural Operator with Learnable Adaptive Weighting (PIFNO-LAW) module.

    This class implements a dual-network architecture for solving PDEs using physics-informed
    neural operators. It consists of:
    - net1: The primary FNO network that predicts the solution field
    - net2: A weighting network that learns to adaptively weight the physics residual

    The training objective combines data loss, initial condition loss, physics loss,
    and a sharpened regularization term for the weighting network.
    """

    net1: torch.nn.Module
    net2: torch.nn.Module

    def __init__(
        self,
        net1: torch.nn.Module,
        net2: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_method,
        compile_model: bool = True,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        lambda_ic: float = 0.25,
        lambda_data: float = 0.25,
        lambda_f: float = 0.25,
        beta_reg: float = 2.5,
        padding: int = 5,
        spatial_padding: int = 0,
        start_blend_ratio: float = 0.05,
        warmup_ratio: float = 0.15,
    ):
        """Initialize the PIFNOLAWModule."""
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net1 = net1
        self.net2 = net2
        self.loss_method = loss_method
        self.padding = padding

    def configure_model(self) -> None:
        """Initialize and compile the neural networks.

        Applies torch.compile optimization to both net1 and net2 if compilation is enabled
        in the hyperparameters. This method is called once during model setup.
        """
        if not self.hparams.compile_model:
            return

        if isinstance(self.net1, torch._dynamo.eval_frame.OptimizedModule):
            return

        self.net1 = cast(
            torch.nn.Module,
            torch.compile(
                self.net1,
                options={"shape_padding": True, "triton.cudagraphs": True},
            ),
        )

        self.net2 = cast(
            torch.nn.Module,
            torch.compile(
                self.net2,
                options={"shape_padding": True, "triton.cudagraphs": True},
            ),
        )
        return super().configure_model()

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the primary FNO network (net1).

        Args:
            input_batch: Input tensor containing the initial condition and coordinates.

        Returns:
            The predicted solution field tensor.

        """
        return self.net1(input_batch)

    def model_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        is_training: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Perform a single model step on a batch of data.

        Computes the total loss as a weighted combination of:
        - Data loss (LpLoss between prediction and ground truth)
        - Initial condition loss (LpLoss at t=0)
        - Physics loss (PDE residual weighted by net2)
        - Sharpened regularization (adaptive weighting for smooth/shock regions)

        Args:
            batch: A tuple containing (input_batch, output) tensors.
            is_training: Whether this step is performed during training (unused, kept for API compatibility).

        Returns:
            A tuple containing:
                - total_loss: Weighted sum of all loss components
                - data_loss: Data matching loss
                - physics_loss: PDE residual loss
                - ic_loss: Initial condition loss
                - loss_reg_weight: Sharpened regularization loss
                - prediction: Model prediction tensor
                - model2_output: Adaptive weighting output from net2

        """
        input_batch, output = batch
        prediction = self._normalize_and_pad_input(input_batch).squeeze(-1)

        # 1. Data Loss
        data_loss = LpLoss()(prediction, output)

        # 2. IC Loss
        initial_condition_gt = output[:, 0, :]
        initial_condition_pred = prediction[:, 0, :]
        ic_loss = LpLoss()(initial_condition_pred, initial_condition_gt)

        # 3. Physics Loss (Net2 weighted)
        max_epochs = self.trainer.max_epochs
        start_blend_epoch = self.hparams.start_blend_ratio * float(
            max_epochs if max_epochs is not None else 1
        )
        warmup_epochs = self.hparams.warmup_ratio * float(
            max_epochs if max_epochs is not None else 1
        )

        blend_ratio = max(
            0.0,
            min(
                1.0,
                (float(self.current_epoch) - start_blend_epoch)
                / (warmup_epochs + 1e-8),
            ),
        )
        # Wrap in a 0-d tensor to avoid re-compiling net2 every epoch
        blend_ratio_tensor = torch.tensor(blend_ratio, device=self.device)

        physics_loss, model2_output, ux = self.loss_method(
            u=prediction,
            dt=self.dt,
            dx=self.dx,
            dy=self.dy,
            input_batch=input_batch,
            output=output,
            net2=self.net2,
            blend_ratio=blend_ratio_tensor,
            beta=self.hparams.beta_reg,
            is_detach=False,
        )

        # 4. Sharpened Regularization (Decoupled per variable)
        with torch.no_grad():
            abs_ux = torch.abs(ux)

            # Normalize spatially per channel
            if abs_ux.ndim == 3:  # 1D Burgers
                normalizer = torch.mean(abs_ux, dim=(2), keepdim=True) + 1e-8
            elif (
                abs_ux.ndim == 4 and abs_ux.shape[-1] == 3
            ):  # 1D Euler [B, T, X, C]
                normalizer = torch.mean(abs_ux, dim=(2), keepdim=True) + 1e-8
            else:
                normalizer = torch.mean(abs_ux, dim=(2, 3), keepdim=True) + 1e-8

            normalized_abs_ux = abs_ux / normalizer

            beta = self.hparams.beta_reg
            weight_smooth = torch.exp(-(beta * normalized_abs_ux))
            weight_shock = 1.0 - weight_smooth

        loss_reg_smooth = (weight_smooth * (model2_output - 1.0)) ** 2
        loss_reg_shock = (weight_shock * (model2_output - 0.0)) ** 2

        loss_reg_weight = torch.mean(loss_reg_smooth + loss_reg_shock)

        total_loss = (
            self.hparams.lambda_ic * ic_loss
            + self.hparams.lambda_f * physics_loss
            + self.hparams.lambda_data * data_loss
            + loss_reg_weight
        )

        return (
            total_loss,
            data_loss,
            physics_loss,
            ic_loss,
            loss_reg_weight,
            prediction,
            model2_output,
        )

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        Computes the total loss and logs individual loss components for monitoring.

        Args:
            batch: A batch of data containing (input, target) tensors.
            batch_idx: The index of the current batch within the epoch.

        Returns:
            The total loss tensor for backpropagation.

        """
        (
            total_loss,
            data_loss,
            physical_loss,
            ic_loss,
            loss_model2,
            prediction,
            model2_output,
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
            {
                "train/data_loss": data_loss,
                "train/physical_loss": physical_loss,
                "train/loss_model2": loss_model2,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return total_loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        Computes and logs validation losses without updating model parameters.

        Args:
            batch: A batch of data containing (input, target) tensors.
            batch_idx: The index of the current batch within the epoch.

        """
        (
            total_loss,
            data_loss,
            physical_loss,
            ic_loss,
            loss_model2,
            prediction,
            model2_output,
        ) = self.model_step(batch=batch, is_training=True)

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
            {
                "val/data_loss": data_loss,
                "val/physical_loss": physical_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the optimizer and learning rate scheduler for training.

        Sets up the optimizer to jointly optimize both net1 (FNO) and net2 (weighting network).
        If a scheduler is provided, it configures epoch-based scheduling with validation loss monitoring.

        Returns:
            A dictionary containing:
                - optimizer: The configured optimizer
                - lr_scheduler: (optional) Learning rate scheduler configuration with:
                    - scheduler: The scheduler instance
                    - monitor: Metric to monitor for scheduling decisions
                    - interval: Scheduling frequency ('epoch')
                    - frequency: How often to apply the scheduler
            If no scheduler is configured, returns only the optimizer.

        """
        optimizer = self.hparams.optimizer(
            params=chain(self.net1.parameters(), self.net2.parameters())
        )

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
