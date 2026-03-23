import h5py
import numpy as np
import torch


class DatasetBuilder:
    def __init__(self, data_dir, sub_x, sub_t):
        try:
            with h5py.File(data_dir, "r") as hf:
                input_data = torch.tensor(np.array(hf["a"]))
                output_data = torch.tensor(np.array(hf["u"]))
                self.x = torch.tensor(np.array(hf["x"]))
                self.time = torch.tensor(np.array(hf["t"]))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Dataset file not found: {data_dir}"
            ) from e
        except KeyError as e:
            raise KeyError(f"Key not found in dataset file: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {e}") from e

        self.sub_x = sub_x
        self.sub_t = sub_t
        self.input_data = input_data[:, ::sub_x]
        self.output_data = output_data[:, ::sub_t, ::sub_x]
        self.sample_x = self.x[::sub_x]
        self.sample_t = self.time[::sub_t]
        self.nx = self.sample_x.shape[0]
        self.t_num = self.sample_t.shape[0]
        self.n_samples = input_data.shape[0]

    def build(self, index):
        pass
