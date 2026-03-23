from typing import Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule

from .components.loss.utils import LpLoss, compute_total_variation


class BaseModule(LightningModule):
    """Base class for PIFNO Modules.
    Encapsulates common methods for training and evaluation.
    """

    padding: int

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if stage == "fit" or stage == "test":
            datamodule = self.trainer.datamodule
            self.input_mean = datamodule.input_mean.to(self.device)
            self.input_std = datamodule.input_std.to(self.device)
            self.output_mean = datamodule.output_mean.to(self.device)
            self.output_std = datamodule.output_std.to(self.device)

            if hasattr(datamodule, "dx"):
                self.dx = datamodule.dx
            if hasattr(datamodule, "dy"):
                self.dy = datamodule.dy
            if hasattr(datamodule, "dt"):
                self.dt = datamodule.dt

    def forward(
        self, input_batch: torch.Tensor
    ) -> torch.Tensor:  # pyrefly: ignore[bad-param-name-override]
        """Subclasses must implement the forward pass."""
        raise NotImplementedError

    def _normalize_and_pad_input(
        self, input_batch: torch.Tensor
    ) -> torch.Tensor:
        input_batch = (input_batch - self.input_mean) / self.input_std

        padding_val: int = self.padding
        spatial_padding_val: int = self.hparams.spatial_padding

        if input_batch.ndim == 4:
            input_batch = F.pad(
                input_batch,
                (
                    0,
                    0,
                    spatial_padding_val,
                    spatial_padding_val,
                    0,
                    padding_val,
                ),
                mode="constant",
                value=0.0,
            )
            prediction = self.forward(input_batch)
            prediction = prediction[
                :,
                : prediction.shape[1] - padding_val,
                spatial_padding_val : prediction.shape[2] - spatial_padding_val,
            ]
        elif input_batch.ndim == 5:
            input_batch = F.pad(
                input_batch,
                (
                    0,
                    0,
                    spatial_padding_val,
                    spatial_padding_val,
                    spatial_padding_val,
                    spatial_padding_val,
                    0,
                    padding_val,
                ),
                mode="constant",
                value=0.0,
            )
            prediction = self.forward(input_batch)
            prediction = prediction[
                :,
                : prediction.shape[1] - padding_val,
                spatial_padding_val : prediction.shape[2] - spatial_padding_val,
                spatial_padding_val : prediction.shape[3] - spatial_padding_val,
            ]
        else:
            raise ValueError(
                "Input batch must be 4D or 5D tensor, got {}".format(
                    input_batch.ndim
                )
            )

        prediction = prediction * self.output_std + self.output_mean
        return prediction

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set."""
        input_batch, output = batch
        prediction = self._normalize_and_pad_input(input_batch)

        if prediction.shape[-1] == 1:
            prediction = prediction.squeeze(-1)
        if output.shape[-1] == 1:
            output = output.squeeze(-1)

        l2_data_loss = LpLoss(p=2)(prediction, output)
        l1_data_loss = LpLoss(p=1)(prediction, output)

        tv_pred_batch = compute_total_variation(prediction)
        tv_target_batch = compute_total_variation(output)

        epsilon = 1e-6
        tv_rel_error_signed = (tv_pred_batch - tv_target_batch) / (
            tv_target_batch + epsilon
        )

        tv_metric_abs = torch.mean(torch.abs(tv_rel_error_signed)).item()

        self.log_dict(
            {
                "test/data_loss": l2_data_loss,
                "test/l1_data_loss": l1_data_loss,
                "test/tv_mae": tv_metric_abs,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.log("last_epoch", float(self.current_epoch))
        pass
