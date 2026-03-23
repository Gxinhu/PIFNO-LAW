from typing import Tuple

import numpy as np
import torch


def _compute_wave_number(
    nx: int, distance: float, device: torch.device
) -> torch.Tensor:
    """Computes the wave number for FFT-based differentiation.

    Args:
        nx (int): Number of grid points in the spatial dimension.
        distance (float): Physical length of the spatial domain.
        device (torch.device): Device to store the wave number tensor.

    Returns:
        torch.Tensor: Wave number tensor of shape (1, 1, nx).

    """
    k_max = nx // 2
    wave_number = torch.cat(
        (
            torch.arange(start=0, end=k_max, step=1, device=device),
            torch.arange(start=-k_max, end=0, step=1, device=device),
        ),
        dim=0,
    ).reshape(1, 1, nx)
    return wave_number * (2j * np.pi / distance)


def _fft_derivative(
    u: torch.Tensor, distance: float, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Computes the first, second, and third derivatives of u using FFT.

    Args:
        u (torch.Tensor): Input tensor of shape (batch_size, time_steps, nx).
        distance (float): Physical length of the spatial domain.
        device (torch.device): Device to perform computations.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: First, second, and third derivatives (ux, uxx, uxxx)
                                                        of shape (batch_size, time_steps, nx).

    """
    nx = u.shape[-1]  # Get nx from the last dimension of u
    u_hat = torch.fft.fft(u, dim=-1)
    wave_number = _compute_wave_number(nx, distance, device)
    wave_number_x = wave_number  # Rename for clarity

    # Compute derivatives in Fourier space
    ux_hat = wave_number_x * u_hat
    uxx_hat = wave_number_x * ux_hat
    uxxx_hat = wave_number_x * uxx_hat if u.ndim > 2 else None
    # Inverse FFT to get derivatives in real space
    ux = torch.fft.irfft(ux_hat[..., : nx // 2 + 1], dim=-1, n=nx)
    uxx = torch.fft.irfft(uxx_hat[..., : nx // 2 + 1], dim=-1, n=nx)
    if uxxx_hat is not None:
        uxxx = torch.fft.irfft(uxxx_hat[..., : nx // 2 + 1], dim=-1, n=nx)
    else:
        uxxx = None

    return ux, uxx, uxxx


class LpLoss(object):
    """loss function with rel/abs Lp loss
    """

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def calculate_local_tv(u, window_size=5):
    """Calculate the local Total Variation (TV) of a 1D array u as a shock indicator,
    optimized for maximum performance.

    Args:
        u (torch.Tensor): 1D tensor, e.g., numerical solution u(x) of Burgers' equation.
        window_size (int): Window size for calculating local TV (suggested to be odd, e.g., 5, 7, ...).

    Returns:
        torch.Tensor: Tensor of the same shape as u, containing local TV value at each position.

    """
    nx = len(u)
    half_window = window_size // 2

    # Pre-compute all absolute differences
    abs_diff = torch.abs(u[1:] - u[:-1])

    # Initialize output tensor
    local_tv = torch.zeros_like(u)

    # Create a summed area table (cumulative sum) for fast window operations
    cumsum_diff = torch.zeros_like(u)
    cumsum_diff[1:] = torch.cumsum(abs_diff, dim=0)

    # Calculate start and end indices for each window
    indices = torch.arange(nx, device=u.device)
    start_indices = torch.clamp(indices - half_window, min=0)
    end_indices = torch.clamp(indices + half_window, max=nx - 1)

    # Create masks for different boundary conditions
    start_is_zero = start_indices == 0
    end_is_edge = end_indices == nx - 1
    normal_windows = ~(start_is_zero | end_is_edge)

    # Handle normal windows (not touching edges)
    normal_mask = normal_windows
    local_tv[normal_mask] = (
        cumsum_diff[end_indices[normal_mask]]
        - cumsum_diff[start_indices[normal_mask] - 1]
    )

    # Handle windows that start at index 0
    left_edge_mask = start_is_zero & ~end_is_edge
    local_tv[left_edge_mask] = cumsum_diff[end_indices[left_edge_mask]]

    # Handle windows that include the right edge but don't start at 0
    right_edge_mask = end_is_edge & ~start_is_zero
    local_tv[right_edge_mask] = (
        cumsum_diff[nx - 1] - cumsum_diff[start_indices[right_edge_mask] - 1]
    )

    # Handle windows that both start at 0 and include the right edge
    both_edges_mask = start_is_zero & end_is_edge
    local_tv[both_edges_mask] = cumsum_diff[nx - 1]

    return local_tv


def compute_total_variation(x: torch.Tensor):
    """Computes the Total Variation (TV) for spatial fields.

    Args:
        x: Tensor of shape [Batch, Time, X] or [Batch, Time, X, Y]

    Returns:
        Tensor of shape [Batch, Time] containing the TV for each sample.

    """
    # Case 1: 1D Spatial Domain [Batch, Time, X]
    if x.ndim == 3:
        # Calculate sum of absolute differences along the spatial dimension
        # Formula: sum(|u_{i+1} - u_i|)
        tv_x = torch.sum(torch.abs(x[..., 1:] - x[..., :-1]), dim=-1)
        return tv_x

    # Case 2: 2D Spatial Domain [Batch, Time, X, Y]
    elif x.ndim == 4:
        # Calculate Anisotropic TV (standard for image/grid tasks)
        # Formula: sum(|u_{x+1} - u_x|) + sum(|u_{y+1} - u_y|)

        # Difference along X-axis
        tv_x = torch.sum(
            torch.abs(x[..., 1:, :] - x[..., :-1, :]), dim=(-2, -1)
        )
        # Difference along Y-axis
        tv_y = torch.sum(
            torch.abs(x[..., :, 1:] - x[..., :, :-1]), dim=(-2, -1)
        )
        return tv_x + tv_y

    else:
        raise ValueError(
            f"Unsupported tensor shape for TV calculation: {x.shape}"
        )
