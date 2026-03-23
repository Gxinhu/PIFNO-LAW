import pytorch_lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint


class ConditionalCheckpoint(ModelCheckpoint):
    """A custom callback that starts saving checkpoints only after a specified epoch.
    """

    def __init__(self, *args, start_after_epoch: int = 0, **kwargs):
        """Args:
        start_after_epoch (int): The epoch after which saving should begin.
                                 For example, if set to 10, the first possible
                                 save will occur at the end of the 11th epoch.

        """
        super().__init__(*args, **kwargs)
        self.start_after_epoch = start_after_epoch

    def on_train_epoch_end(  # pyrefly: ignore[bad-param-name-override]
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # Check if the current epoch has reached our specified threshold
        if trainer.current_epoch < self.start_after_epoch:
            # If not, skip and do nothing
            return

        # If the threshold is reached, call the parent class's (ModelCheckpoint)
        # original method to execute all normal checks and saving logic
        # (e.g., checking if val_loss has improved)
        # pyrefly: ignore[bad-param-name-override, missing-argument, bad-argument-type]
        super().on_train_epoch_end(trainer, pl_module)

    # If you primarily trigger saving based on the validation set (monitor='val_loss'),
    # overriding on_validation_end would be more robust.
    def on_validation_end(  # pyrefly: ignore[bad-param-name-override]
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.current_epoch < self.start_after_epoch:
            return
        # pyrefly: ignore[bad-param-name-override, missing-argument, bad-argument-type]
        super().on_validation_end(trainer, pl_module)
