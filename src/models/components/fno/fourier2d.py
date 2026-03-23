import torch
import torch.nn as nn
import torch.nn.functional as F

from .basics import SpectralConv2d


class FNN2d(nn.Module):
    def __init__(
        self,
        modes1,
        modes2,
        width=64,
        fc_dim=128,
        layers=None,
        in_dim=3,
        out_dim=1,
        activation="tanh",
        pad_x=0,
        pad_y=0,
        last_activation=False,
    ):
        super(FNN2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.padding = (0, 0, 0, pad_y, 0, pad_x)
        # input channel is 3: (a(x, y), x, y)
        if layers is None:
            layers = [width] * 4
        self.layers = layers
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList([
            SpectralConv2d(in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num in zip(
                self.layers, self.layers[1:], self.modes1, self.modes2
            )
        ])

        self.ws = nn.ModuleList([
            nn.Conv1d(in_size, out_size, 1)
            for in_size, out_size in zip(self.layers, self.layers[1:])
        ])

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        self.last_activation = last_activation
        if activation == "tanh":
            self.activation = F.tanh
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "leaky":
            self.activation = F.leaky_relu
        else:
            raise ValueError(f"{activation} is not supported")

    def forward(self, x) -> torch.Tensor:
        """Args:
            - x : (batch size, x_grid, y_grid, 2)

        Returns:
            - x: (batch size, x_grid, y_grid, 1)

        """
        length = len(self.ws)
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(
                batchsize, self.layers[i + 1], size_x, size_y
            )
            x = x1 + x2
            if i != length - 1:
                x = self.activation(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x.reshape(batchsize, size_x, size_y, self.out_dim)
        if self.last_activation:
            x = F.sigmoid(x)
        return x

