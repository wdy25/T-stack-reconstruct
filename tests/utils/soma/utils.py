import torch
import numpy as np
from typing import Tuple, Optional

from core.ir.data import DataType


# --- Meanpooling ---
def make_input_payload_for_meanpooling(shape: Tuple[int, int, int, int]) -> torch.Tensor:
    torch.manual_seed(2)
    x = torch.randn(shape, dtype=torch.bfloat16)
    x[0, 0, 0, 0] = torch.tensor(-48.0, dtype=torch.bfloat16)
    x[0, 0, 0, 1] = torch.tensor(48.0, dtype=torch.bfloat16)
    x[0, 0, 0, 2] = torch.tensor(-0.0, dtype=torch.bfloat16)
    x[0, 0, 0, 3] = torch.tensor(0.0, dtype=torch.bfloat16)
    return x

# Expected output is computed via PyTorch's conv2d. It's a little different from the result of meanpooling, because the BF16 accumulation order of these two is not exactly the same.  
def compute_expected_torch_meanpooling(
    x: torch.Tensor,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    output_dtype: DataType,
    scaler_mode: int,
    scaler: int,
) -> torch.Tensor:
    kernel_h, kernel_w = kernel_size
    c_in = x.shape[-1]

    x_nchw = x.permute(0, 3, 1, 2)
    weight = torch.ones((c_in, 1, kernel_h, kernel_w), dtype=torch.bfloat16)
    y_nchw = torch.nn.functional.conv2d(
        x_nchw,
        weight,
        bias=None,
        stride=stride,
        padding=0,
        groups=c_in,
    )
    y = y_nchw.permute(0, 2, 3, 1)

    if scaler_mode == 1:
        y *= 2 ** scaler
    elif scaler_mode != 0:
        raise ValueError(f"Unsupported scaler_mode: {scaler_mode}")

    if output_dtype == DataType.BF16:
        return y.to(torch.bfloat16)
    if output_dtype == DataType.INT8:
        return y.clamp(-128, 127).to(torch.int8)

    raise ValueError(f"Unsupported output_dtype: {output_dtype}")

# --- Maxpooling ---
def make_input_payload_for_maxpooling(shape: Tuple[int, int, int, int]) -> torch.Tensor:
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    x[0, 0, 0, 0] = torch.tensor(-160.0, dtype=torch.bfloat16)
    x[0, 0, 0, 1] = torch.tensor(160.0, dtype=torch.bfloat16)
    x[0, 0, 0, 2] = torch.tensor(0.0, dtype=torch.bfloat16)
    x[0, 0, 0, 3] = torch.tensor(-0.0, dtype=torch.bfloat16)
    return x

def make_bias_vector_for_maxpooling(c: int) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn((1, 1, 1, c), dtype=torch.bfloat16)

def compute_expected_torch_maxpooling(
    x: torch.Tensor,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    output_dtype: DataType,
    bias_mode: int,
    scaler_mode: int,
    scaler: int,
    pool_type: str,
    bias_scalar: Optional[float] = None,
    bias_vector: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    pool2d = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
    x_nchw = x.permute(0, 3, 1, 2) # NHWC to NCHW for PyTorch pooling
    if pool_type == "max":
        y_nchw = pool2d(x_nchw)
    elif pool_type == "min":
        y_nchw = -pool2d(-x_nchw)

    y = y_nchw.permute(0, 2, 3, 1)

    if bias_mode == 1:
        bias_tensor = torch.tensor(bias_scalar, dtype=torch.bfloat16).reshape(1, 1, 1, 1)
        if pool_type == "max":
            y = torch.maximum(y, bias_tensor)
        else:
            y = torch.minimum(y, bias_tensor)
    elif bias_mode == 2:
        if pool_type == "max":
            y = torch.maximum(y, bias_vector)
        else:
            y = torch.minimum(y, bias_vector)
    elif bias_mode != 0:
        raise ValueError(f"Unsupported bias_mode: {bias_mode}")

    if scaler_mode == 1:
        y *= 2 ** scaler

    if output_dtype == DataType.BF16:
        return y.to(torch.bfloat16)
    if output_dtype == DataType.INT8:
        return y.clamp(-128, 127).to(torch.int8)
    raise ValueError(f"Unsupported output_dtype: {output_dtype}")


# --- Nonlinear ---
def make_input_payload_for_nonlinear(func_name: str, shape: tuple[int, ...]) -> torch.Tensor:
    # Generate BF16 input. Keep values in safe ranges and avoid NaNs for sqrt.
    x = torch.randn(shape, dtype=torch.bfloat16)

    func = func_name.lower()
    if func == "sqrt":
        x = x.abs()
        # Force some exact values
        x_flat = x.reshape(-1)
        if x_flat.numel() >= 6:
            x_flat[0] = 0.0
            x_flat[1] = 1.0
            x_flat[2] = 4.0
            x_flat[3] = 16.0
            x_flat[4] = 0.25
            x_flat[5] = 2.0
        x = x_flat.reshape(shape)
    elif func == "exp":
        x = x.clamp(-5.0, 5.0)
        x_flat = x.reshape(-1)
        if x_flat.numel() >= 6:
            x_flat[0] = -5.0
            x_flat[1] = -1.0
            x_flat[2] = -0.0
            x_flat[3] = 0.0
            x_flat[4] = 1.0
            x_flat[5] = 5.0
        x = x_flat.reshape(shape)
    elif func in ("sin", "cos"):
        # sin/cos: keep range moderate for stable BF16 behavior
        x = x.clamp(-2.0, 2.0)
        x_flat = x.reshape(-1)
        if x_flat.numel() >= 6:
            x_flat[0] = -2.0
            x_flat[1] = -1.0
            x_flat[2] = -0.0
            x_flat[3] = 0.0
            x_flat[4] = 1.0
            x_flat[5] = 2.0
        x = x_flat.reshape(shape)
    else:
        raise ValueError(f"Unsupported function for test: {func_name}")

    return x

def compute_expected_torch_nonlinear(x: torch.Tensor, func_name: str) -> torch.Tensor:
    func = func_name.lower()

    if func == "sqrt":
        y = torch.sqrt(x)
    elif func == "sin":
        # Match PrimNonlinear: sin(pi * a)
        y = torch.sin(np.pi * x)
    elif func == "cos":
        # Match PrimNonlinear: cos(pi * a)
        y = torch.cos(np.pi * x)
    elif func == "exp":
        y = torch.exp(x)
    else:
        raise ValueError(f"Unsupported function for test: {func_name}")

    return y.to(torch.bfloat16)

# --- Convert ---
def make_input_payload_for_convert(dtype: DataType, shape: tuple[int, int]) -> torch.Tensor:
    if dtype == DataType.INT8:
        x = torch.randint(-128, 127, shape, dtype=torch.int8)
        # Force some edge values for INT8->BIN and clamp coverage
        x[0, 0] = -128
        x[0, 1] = -1
        x[0, 2] = 0
        x[0, 3] = 1
        x[0, 4] = 127
        return x

    if dtype == DataType.BF16:
        x = torch.randn(shape).to(torch.bfloat16)
        # Include +/-0.0 and some extremes for BF16->INT8 and BF16->BIN
        x[0, 0] = torch.tensor(-0.0, dtype=torch.bfloat16)
        x[0, 1] = torch.tensor(0.0, dtype=torch.bfloat16)
        x[0, 2] = torch.tensor(-129.0, dtype=torch.bfloat16)
        x[0, 3] = torch.tensor(128.0, dtype=torch.bfloat16)
        x[0, 4] = torch.tensor(-1.0, dtype=torch.bfloat16)
        x[0, 5] = torch.tensor(1.0, dtype=torch.bfloat16)
        return x

    if dtype == DataType.SPIKE:
        x = torch.randint(0, 2, shape, dtype=torch.int64).to(torch.bool)
        x[0, 0] = False
        x[0, 1] = True
        return x

    raise ValueError(f"Unsupported dtype for payload: {dtype}")

def compute_expected_torch_convert(x: torch.Tensor, in_dtype: DataType, out_dtype: DataType) -> torch.Tensor:
    if in_dtype == out_dtype:
        raise ValueError("Convert operation requires differing dtypes.")

    if in_dtype == DataType.INT8 and out_dtype == DataType.BF16:
        return x.to(torch.bfloat16)

    if in_dtype == DataType.BF16 and out_dtype == DataType.INT8:
        return x.clamp(-128, 127).to(torch.int8)

    if in_dtype == DataType.INT8 and out_dtype == DataType.SPIKE:
        return (x > 0).to(torch.bool)

    if in_dtype == DataType.BF16 and out_dtype == DataType.SPIKE:
        # Match PrimConvert: check BF16 sign bit via int16 view, includes -0.0 as negative.
        return (x.view(torch.int16) > 0).to(torch.bool)

    if in_dtype == DataType.SPIKE and out_dtype == DataType.INT8:
        return x.to(torch.int8)

    if in_dtype == DataType.SPIKE and out_dtype == DataType.BF16:
        return torch.where(
            x,
            torch.tensor(1.0, dtype=torch.bfloat16),
            torch.tensor(-0.0, dtype=torch.bfloat16),
        )

    raise ValueError(f"Unsupported conversion from {in_dtype} to {out_dtype}.")


# --- LUT ---
def make_input_payload_for_lut(dtype: DataType, shape: tuple[int, int]) -> torch.Tensor:
    if dtype == DataType.INT8:
        x = torch.randint(-128, 127, shape, dtype=torch.int8)
        x[0, 0] = -128
        x[0, 1] = -1
        x[0, 2] = 0
        x[0, 3] = 1
        x[0, 4] = 127
        return x

    if dtype == DataType.BF16:
        x = torch.randn(shape).to(torch.bfloat16)
        x[0, 0] = torch.tensor(-0.0, dtype=torch.bfloat16)
        x[0, 1] = torch.tensor(0.0, dtype=torch.bfloat16)
        x[0, 2] = torch.tensor(-1.0, dtype=torch.bfloat16)
        x[0, 3] = torch.tensor(1.0, dtype=torch.bfloat16)
        x[0, 4] = torch.tensor(-8.0, dtype=torch.bfloat16)
        x[0, 5] = torch.tensor(8.0, dtype=torch.bfloat16)
        return x

    raise ValueError(f"Unsupported dtype for LUT payload: {dtype}")


def compute_expected_torch_lut(x: torch.Tensor, out_dtype: DataType, function: str) -> torch.Tensor:
    func = function.lower()
    x_fp32 = x.to(torch.float32)

    if func == "sigmoid":
        y_fp32 = torch.sigmoid(x_fp32)
    elif func == "tanh":
        y_fp32 = torch.tanh(x_fp32)
    elif func == "relu":
        y_fp32 = torch.relu(x_fp32)
    elif func == "gelu":
        y_fp32 = torch.nn.functional.gelu(x_fp32)
    else:
        raise ValueError(f"Unsupported LUT function for test: {function}")

    if out_dtype == DataType.INT8:
        return y_fp32.round().clamp(-128, 127).to(torch.int8)
    if out_dtype == DataType.BF16:
        return y_fp32.to(torch.bfloat16)

    raise ValueError(f"Unsupported LUT output dtype for test: {out_dtype}")