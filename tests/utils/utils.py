import torch

def relative_error(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> float:
    """Mean relative error: mean(|a-b|) / (mean(|b|) + eps)."""

    num = torch.mean(torch.abs(a - b), dtype=torch.bfloat16)
    den = torch.mean(torch.abs(b), dtype=torch.bfloat16) + eps
    return (num / den).item()