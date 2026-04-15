import math

import torch
import torch.nn.functional as F


def compute_distance(F_tokens: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Compute min cosine distance from each token to a prototype set.

    Args:
        F_tokens: Tensor of shape [B, N, C] or [N, C].
        prototypes: Tensor of shape [K, C] or [B, K, C].

    Returns:
        Tensor of shape [B, N] or [N] with min cosine distance.
    """
    squeeze_batch = False
    if F_tokens.dim() == 2:
        F_tokens = F_tokens.unsqueeze(0)
        squeeze_batch = True

    if prototypes.dim() == 2:
        prototypes = prototypes.unsqueeze(0).expand(F_tokens.shape[0], -1, -1)

    F_norm = F.normalize(F_tokens, dim=-1)
    P_norm = F.normalize(prototypes, dim=-1)
    sim = torch.matmul(F_norm, P_norm.transpose(1, 2))
    dist = 1.0 - sim.max(dim=-1)[0]

    if squeeze_batch:
        return dist.squeeze(0)
    return dist


def compute_alpha(P_in: torch.Tensor, min_alpha: float = 0.2, max_alpha: float = 0.8) -> torch.Tensor:
    """Compute adaptive fusion weight from intrinsic prototype variance."""
    if P_in.dim() == 2:
        var = torch.var(P_in, dim=0).mean()
        alpha = torch.exp(-var)
        return torch.clamp(alpha, min_alpha, max_alpha)

    var = torch.var(P_in, dim=1).mean(dim=1)
    alpha = torch.exp(-var)
    return torch.clamp(alpha, min_alpha, max_alpha)


def fuse_distances(
    D_in: torch.Tensor,
    D_pr: torch.Tensor,
    mode: str = "hybrid",
    alpha_mode: str = "adaptive",
    alpha_fixed: float = 0.5,
    P_in: torch.Tensor = None,
):
    """Fuse intrinsic and prior distances for inp/prior/hybrid modes."""
    mode = mode.lower()
    if mode == "inp":
        alpha = torch.tensor(1.0, device=D_in.device, dtype=D_in.dtype)
        return D_in, alpha
    if mode == "prior":
        alpha = torch.tensor(0.0, device=D_pr.device, dtype=D_pr.dtype)
        return D_pr, alpha

    if alpha_mode == "fixed":
        alpha = torch.tensor(alpha_fixed, device=D_in.device, dtype=D_in.dtype)
    else:
        if P_in is None:
            raise ValueError("P_in must be provided when alpha_mode='adaptive'.")
        alpha = compute_alpha(P_in)

    if alpha.dim() == 0:
        D_final = alpha * D_in + (1.0 - alpha) * D_pr
    else:
        D_final = alpha.unsqueeze(-1) * D_in + (1.0 - alpha.unsqueeze(-1)) * D_pr
    return D_final, alpha


def distances_to_map(distances: torch.Tensor) -> torch.Tensor:
    """Convert [B, N] token distances into [B, 1, H, W] map."""
    if distances.dim() == 1:
        distances = distances.unsqueeze(0)
    side = int(math.sqrt(distances.shape[1]))
    if side * side != distances.shape[1]:
        raise ValueError(f"Token count {distances.shape[1]} is not a perfect square.")
    return distances.view(distances.shape[0], 1, side, side).contiguous()
