"""Loss functions for the 1D Burgers' equation."""

import torch
import torch.nn.functional as F


def inviscid_burgers(
    u: torch.Tensor,
    dx: float,
    dt: float,
    alpha1: float = 1.0,
    beta1: float = 1.0,
    is_weighted: bool = True,
    **kwargs,
) -> torch.Tensor:
    """Compute the Burgers' equation residual using a heuristic weight.

    Args:
        u (torch.Tensor): Solution field u(x, t) of shape (batch_size, time_steps, nx).
        dx (float): Grid spacing in the spatial dimension.
        dt (float): Time step size.
        alpha1 (float): Hyperparameter alpha.
        beta1 (float): Hyperparameter beta.
        is_weighted (bool): Whether to apply heuristic weighting.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: The mean squared error loss of the PDE residual.

    """
    _, nt, nx = u.shape

    ut = (u[:, 1:, :] - u[:, :-1, :]) / dt

    k = 2.0 * torch.pi * torch.fft.rfftfreq(nx, d=dx)
    k = k.reshape(1, 1, -1).to(u.device)
    u_hat = torch.fft.rfft(u, dim=-1)
    ux_hat = 1j * k * u_hat
    ux = torch.fft.irfft(ux_hat, n=nx, dim=-1)
    residual_pde = ut + u[:, :-1] * ux[:, :-1]
    if is_weighted:
        weight = 1 / (alpha1 * (torch.abs(ux)) ** beta1 + 1)
        residual_pde = residual_pde[:, 1:] * weight[:, 1:-1, :]
    else:
        residual_pde = residual_pde[:, 1:]
    return F.mse_loss(residual_pde, torch.zeros_like(residual_pde))


def inviscid_burgers_fft_weight(
    u: torch.Tensor,
    dx: float,
    dt: float,
    net2: torch.nn.Module,
    **kwargs,
):
    """Compute the Burgers' equation residual using a neural network learned weight.

    Args:
        u (torch.Tensor): Solution field u(x, t) of shape (batch_size, time_steps, nx).
        dx (float): Grid spacing in the spatial dimension.
        dt (float): Time step size.
        net2 (torch.nn.Module): The auxiliary neural network for the weighting field.
        **kwargs: Must contain 'input_batch' and optionally 'blend_ratio'.

    Returns:
        tuple: A tuple containing the PDE loss, correction net output, and spatial derivative.

    """
    _, _, nx = u.shape
    ut = (u[:, 1:, :] - u[:, :-1, :]) / dt

    k = 2.0 * torch.pi * torch.fft.rfftfreq(nx, d=dx)
    k = k.reshape(1, 1, -1).to(u.device)
    u_hat = torch.fft.rfft(u, dim=-1)
    ux_hat = 1j * k * u_hat
    ux = torch.fft.irfft(ux_hat, n=nx, dim=-1)
    input_batch = kwargs["input_batch"]
    abs_ux = torch.abs(ux)

    residual_pde = ut + u[:, :-1] * ux[:, :-1]  # [B, T-1, nx]
    # Pad residual at t=0 with zeros: the PDE residual is undefined at the IC step.
    # Shape after pad: [B, T, nx] — aligns with u, abs_ux, and coord features.
    residual_padded = torch.cat(
        [torch.zeros_like(residual_pde[:, :1, :]), residual_pde], dim=1
    )
    input_to_net2 = torch.stack(
        (
            u,
            residual_padded,
            ux,
            input_batch[..., 1],
            input_batch[..., 2],
        ),
        dim=-1,
    )
    correction_net = net2(input_to_net2).squeeze(-1)

    blend_ratio = kwargs.get("blend_ratio", 1.0)

    normalizer = torch.max(abs_ux) + 1e-8
    normalized_abs_ux = abs_ux / normalizer
    weight_analytical = torch.exp(-1.0 * normalized_abs_ux)

    correction_pde = (
        1.0 - blend_ratio
    ) * weight_analytical + blend_ratio * correction_net

    pde_term_for_loss = residual_pde[:, 1:] * correction_pde[:, 1:-1]
    loss_pde = F.mse_loss(
        pde_term_for_loss, torch.zeros_like(pde_term_for_loss)
    )
    return loss_pde, correction_net, ux
