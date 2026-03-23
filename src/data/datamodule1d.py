import math
from typing import Any, Dict, Optional

import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class DataModule1D(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        n_train: int = 32,
        n_test: int = 16,
        batch_size: int = 64,
        data_builder: Optional[Any] = None,
        val_ratio: float = 0.2,
        sub_x=1,
        sub_t=1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.n_train = n_train
        self.n_test = n_test
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self.dx = 0
        self.dy = 0

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        self.builder = self.hparams.data_builder()
        self.test_builder = self.hparams.data_builder(sub_x=1, sub_t=1)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            total_samples = self.builder.n_samples
            requested_samples = self.n_train + self.n_test

            if requested_samples > total_samples:
                raise ValueError(
                    f"Requested samples ({requested_samples}) exceed total samples ({total_samples})"
                )

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.split_dataset()

    def split_dataset(self):
        val_size = math.ceil(self.n_train * self.hparams.val_ratio)
        train_val_size = self.n_train + val_size
        indices = np.arange(self.builder.n_samples)[:train_val_size]
        np.random.shuffle(indices)
        train_indices = indices[: self.n_train]
        val_indices = indices[self.n_train :]
        test_indices = np.arange(self.builder.n_samples)[-self.n_test :]
        self.data_train = self.builder.build(train_indices)
        self.data_val = self.builder.build(val_indices)
        self.data_test = self.test_builder.build(test_indices)
        self.input_mean = self.data_train.tensors[0].mean(dim=[0, 1, 2])
        self.input_std = self.data_train.tensors[0].std(dim=[0, 1, 2])
        self.dx = (
            self.data_train.tensors[0][0, 0, 1, -2]
            - self.data_train.tensors[0][0, 0, 0, -2]
        )
        self.dt = (
            self.data_train.tensors[0][0, 1, 0, -1]
            - self.data_train.tensors[0][0, 0, 1, -1]
        )

        self.output_mean = self.data_train.tensors[1].mean(dim=[0, 1, 2])
        self.output_std = self.data_train.tensors[1].std(dim=[0, 1, 2])

        # Avoid division by zero - use mask-based approach for pyrefly compatibility
        zero_std_mask = self.input_std == 0
        self.input_std = self.input_std.clone()
        self.input_std[zero_std_mask] = 1.0
        self.input_mean = self.input_mean.clone()
        self.input_mean[zero_std_mask] = 0.0

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,  # pyrefly: ignore[bad-argument-type]
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,  # pyrefly: ignore[bad-argument-type]
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,  # pyrefly: ignore[bad-argument-type]
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = DataModule1D()
