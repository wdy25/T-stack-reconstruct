import torch

from core.ir.data import DataType

# --- Transposition ---
def make_input_payload_for_transposition(in_dtype: DataType, shape: tuple[int, int, int, int]) -> torch.Tensor:
    if in_dtype == DataType.INT8:
        x = torch.randint(-128, 127, shape, dtype=torch.int8)
        return x
    elif in_dtype == DataType.BF16:
        x = torch.randn(shape, dtype=torch.bfloat16)
        return x

def compute_expected_torch_transposition(x: torch.Tensor, transpose_order: str) -> torch.Tensor:
    if transpose_order == "AB":
        return x.permute(1, 0, 2, 3).contiguous()
    if transpose_order == "AC":
        return x.permute(2, 1, 0, 3).contiguous()
    if transpose_order == "AD":
        return x.permute(3, 1, 2, 0).contiguous()
    if transpose_order == "BC":
        return x.permute(0, 2, 1, 3).contiguous()
    if transpose_order == "BD":
        return x.permute(0, 3, 2, 1).contiguous()
    if transpose_order == "CD":
        return x.permute(0, 1, 3, 2).contiguous()
    raise ValueError(f"Unknown transpose_order: {transpose_order}")