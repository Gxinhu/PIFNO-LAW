import torch

from .dataset import DatasetBuilder


class BurgersDatasetBuilder(DatasetBuilder):
    def __init__(self, data_dir, sub_x, sub_t):
        super().__init__(data_dir, sub_x, sub_t)

    def build(self, index):
        n_sample = index.shape[0]
        input_data = self.input_data[index]
        output_data = self.output_data[index]
        gridx = self.sample_x.reshape(1, 1, self.nx)
        gridt = self.sample_t.reshape(1, self.t_num, 1)
        input_data = input_data.reshape(n_sample, 1, self.nx).repeat([
            1,
            self.t_num,
            1,
        ])
        input_data = torch.stack(
            [
                input_data,
                gridx.repeat([n_sample, self.t_num, 1]),
                gridt.repeat([n_sample, 1, self.nx]),
            ],
            dim=3,
        )
        return torch.utils.data.TensorDataset(
            input_data.float(),
            output_data.float()
        )
