import torch, os
import torch.nn.functional as F

def handwritten_matmul(x_data_in: torch.Tensor, w_data_in: torch.Tensor, bias_data_in: torch.Tensor) -> torch.Tensor:
    """Route to the correct handwritten matmul implementation.

    Supported combinations:
        - x int8, w int8, bias int32  -> output int32 -> bf16
        - x spike(bool), w int8, bias int32 -> output int32 -> bf16
        - x bf16, w bf16, bias bf16 -> output bf16
        - x spike(bool), w bf16, bias bf16 -> output bf16
        - x spike(bool), w spike(bool), bias int32 -> output int32 -> bf16
    """

    x_dtype = x_data_in.dtype
    w_dtype = w_data_in.dtype
    b_dtype = bias_data_in.dtype

    is_spike_x = x_dtype == torch.bool
    is_spike_w = w_dtype == torch.bool

    # ANN int8
    if x_dtype == torch.int8 and w_dtype == torch.int8 and b_dtype == torch.int32:
        return ann_int8_matmul(x_data_in, w_data_in, bias_data_in)

    # SNN spike x + int8 w
    if is_spike_x and w_dtype == torch.int8 and b_dtype == torch.int32:
        return snn_int8_matmul(x_data_in, w_data_in, bias_data_in)

    # ANN bf16
    if x_dtype == torch.bfloat16 and w_dtype == torch.bfloat16 and b_dtype == torch.bfloat16:
        return ann_bf16_matmul(x_data_in, w_data_in, bias_data_in)

    # SNN spike x + bf16 w
    if is_spike_x and w_dtype == torch.bfloat16 and b_dtype == torch.bfloat16:
        return snn_bf16_matmul(x_data_in, w_data_in, bias_data_in)

    # SNN spike x + spike w
    if is_spike_x and is_spike_w and b_dtype == torch.int32:
        return snn_spike_matmul(x_data_in, w_data_in, bias_data_in)

    raise ValueError(
        "Unsupported matmul dtype combination: "
        f"x={x_dtype}, w={w_dtype}, bias={b_dtype}. "
        "Supported: (int8,int8,int32), (bool,int8,int32), (bf16,bf16,bf16), (bool,bf16,bf16), (bool,bool,int32)."
    )

def snn_spike_matmul(x_data_in, w_data_in, bias_data_in):
    """
    SNN SPIKE matmul: spike input, spike weight, int32 bias -> bf16 output.

    Input shape: (dim_A, batch, cin) - SPIKE (bool)
    Weight shape: (dim_A, cin, cout) - SPIKE (bool)
    Bias shape: (cout,) - int32
    Output shape: (dim_A, batch, cout) - bf16

    Note: Accumulates in int32, then casts to bf16.
    """
    dim_A, batch, cin = x_data_in.shape
    dim_A2, cin2, cout = w_data_in.shape
    assert (dim_A == dim_A2) and (cin2 == cin), "Input and weight shape mismatch"
    assert cout % 32 == 0, f"For spike weight matmul, cout must be a multiple of 32, got {cout}"

    cin_res = cin % 256
    cin_loop = cin // 256 if cin_res == 0 else cin // 256 + 1
    cout_loop = cout // 32
    bs_res = batch % 16
    bs_loop = batch // 16 if (batch % 16 == 0) else (batch // 16 + 1)
    y_data_out = torch.zeros((dim_A, batch, cout), dtype=torch.int32)

    for dim in range(dim_A):
        for b0 in range(bs_loop):
            for coutl0 in range(cout_loop):
                for i in range(32):
                    if (b0 < (bs_loop - 1) or bs_res == 0):
                        for j in range(16):
                            y_data_out[dim, b0 * 16 + j, coutl0 * 32 + i] = bias_data_in[coutl0 * 32 + i].to(torch.int32)
                    else:
                        for j in range(bs_res):
                            y_data_out[dim, b0 * 16 + j, coutl0 * 32 + i] = bias_data_in[coutl0 * 32 + i].to(torch.int32)
                for cinl0 in range(cin_loop):
                    for i in range(32):
                        if (cinl0 < (cin_loop - 1) or cin_res == 0):
                            for k in range(256):
                                if (b0 < (bs_loop - 1) or bs_res == 0):
                                    for j in range(16):
                                        y_data_out[dim, b0 * 16 + j, coutl0 * 32 + i] += x_data_in[dim, b0 * 16 + j, cinl0 * 256 + k].to(torch.int32) * w_data_in[dim, cinl0 * 256 + k, coutl0 * 32 + i].to(torch.int32)
                                else:
                                    for j in range(bs_res):
                                        y_data_out[dim, b0 * 16 + j, coutl0 * 32 + i] += x_data_in[dim, b0 * 16 + j, cinl0 * 256 + k].to(torch.int32) * w_data_in[dim, cinl0 * 256 + k, coutl0 * 32 + i].to(torch.int32)
                        else:
                            for k in range(cin_res):
                                if (b0 < (bs_loop - 1) or bs_res == 0):
                                    for j in range(16):
                                        y_data_out[dim, b0 * 16 + j, coutl0 * 32 + i] += x_data_in[dim, b0 * 16 + j, cinl0 * 256 + k].to(torch.int32) * w_data_in[dim, cinl0 * 256 + k, coutl0 * 32 + i].to(torch.int32)
                                else:
                                    for j in range(bs_res):
                                        y_data_out[dim, b0 * 16 + j, coutl0 * 32 + i] += x_data_in[dim, b0 * 16 + j, cinl0 * 256 + k].to(torch.int32) * w_data_in[dim, cinl0 * 256 + k, coutl0 * 32 + i].to(torch.int32)

    return y_data_out.to(torch.bfloat16)

def snn_int8_matmul(x_data_in, w_data_in, bias_data_in):
    """
    SNN INT8 matmul: spike input, int8 weight, int32 bias -> bf16 output.

    Input shape: (dim_A, batch, cin) - SPIKE (bool)
    Weight shape: (dim_A, cin, cout) - int8
    Bias shape: (cout,) - int32
    Output shape: (dim_A, batch, cout) - bf16

    Note: Accumulates in int32, then casts to bf16.
    """
    dim_A, batch, cin = x_data_in.shape
    dim_A2, cin2, cout = w_data_in.shape
    assert (dim_A == dim_A2) and (cin2 == cin), "Input and weight shape mismatch"
    assert cout % 32 == 0, f"For int8 weight matmul, cout must be a multiple of 32, got {cout}"

    cin_res = cin % 256
    cin_loop = cin // 256 if cin_res == 0 else cin // 256 + 1
    cout_loop = cout // 32
    bs_res = batch % 16
    bs_loop = batch // 16 if (batch % 16 == 0) else (batch // 16 + 1)
    y_data_out = torch.zeros((dim_A, batch, cout), dtype=torch.int32)

    for dim in range(dim_A):
        for b0 in range(bs_loop):
            for coutl0 in range(cout_loop):
                for i in range(32):
                    if (b0 < (bs_loop - 1) or bs_res == 0):
                        for j in range(16):
                            y_data_out[dim, b0 * 16 + j, coutl0 * 32 + i] = bias_data_in[coutl0 * 32 + i].to(torch.int32)
                    else:
                        for j in range(bs_res):
                            y_data_out[dim, b0 * 16 + j, coutl0 * 32 + i] = bias_data_in[coutl0 * 32 + i].to(torch.int32)
                for cinl0 in range(cin_loop):
                    for i in range(32):
                        if (cinl0 < (cin_loop - 1) or cin_res == 0):
                            for k in range(256):
                                if (b0 < (bs_loop - 1) or bs_res == 0):
                                    for j in range(16):
                                        y_data_out[dim, b0 * 16 + j, coutl0 * 32 + i] += x_data_in[dim, b0 * 16 + j, cinl0 * 256 + k].to(torch.int32) * w_data_in[dim, cinl0 * 256 + k, coutl0 * 32 + i].to(torch.int32)
                                else:
                                    for j in range(bs_res):
                                        y_data_out[dim, b0 * 16 + j, coutl0 * 32 + i] += x_data_in[dim, b0 * 16 + j, cinl0 * 256 + k].to(torch.int32) * w_data_in[dim, cinl0 * 256 + k, coutl0 * 32 + i].to(torch.int32)
                        else:
                            for k in range(cin_res):
                                if (b0 < (bs_loop - 1) or bs_res == 0):
                                    for j in range(16):
                                        y_data_out[dim, b0 * 16 + j, coutl0 * 32 + i] += x_data_in[dim, b0 * 16 + j, cinl0 * 256 + k].to(torch.int32) * w_data_in[dim, cinl0 * 256 + k, coutl0 * 32 + i].to(torch.int32)
                                else:
                                    for j in range(bs_res):
                                        y_data_out[dim, b0 * 16 + j, coutl0 * 32 + i] += x_data_in[dim, b0 * 16 + j, cinl0 * 256 + k].to(torch.int32) * w_data_in[dim, cinl0 * 256 + k, coutl0 * 32 + i].to(torch.int32)

    return y_data_out.to(torch.bfloat16)

def snn_bf16_matmul(x_data_in, w_data_in, bias_data_in):
    """
    SNN BF16 matmul: spike input, bf16 weight, bf16 bias -> bf16 output.

    Input shape: (dim_A, batch, cin) - SPIKE (bool)
    Weight shape: (dim_A, cin, cout) - bf16
    Bias shape: (cout,) - bf16
    Output shape: (dim_A, batch, cout) - bf16
    """
    dim_A, batch, cin = x_data_in.shape
    dim_A2, cin2, cout = w_data_in.shape
    assert (dim_A == dim_A2) and (cin2 == cin), "Input and weight shape mismatch"
    assert cout % 16 == 0, f"For bf16 weight matmul, cout must be a multiple of 16, got {cout}"

    cin_res = cin % 256
    cin_loop = cin // 256 if cin_res == 0 else cin // 256 + 1
    cout_loop = cout // 16
    bs_res = batch % 8
    bs_loop = batch // 8 if (batch % 8 == 0) else (batch // 8 + 1)
    y_data_out = torch.zeros((dim_A, batch, cout), dtype=torch.bfloat16)

    for dim in range(dim_A):
        for b0 in range(bs_loop):
            for coutl0 in range(cout_loop):
                for i in range(16):
                    if (b0 < (bs_loop - 1) or bs_res == 0):
                        for j in range(8):
                            y_data_out[dim, b0 * 8 + j, coutl0 * 16 + i] = bias_data_in[coutl0 * 16 + i]
                    else:
                        for j in range(bs_res):
                            y_data_out[dim, b0 * 8 + j, coutl0 * 16 + i] = bias_data_in[coutl0 * 16 + i]
                for cinl0 in range(cin_loop):
                    for i in range(16):
                        if (cinl0 < (cin_loop - 1) or cin_res == 0):
                            for k in range(256):
                                if (b0 < (bs_loop - 1) or bs_res == 0):
                                    for j in range(8):
                                        y_data_out[dim, b0 * 8 + j, coutl0 * 16 + i] += x_data_in[dim, b0 * 8 + j, cinl0 * 256 + k] * w_data_in[dim, cinl0 * 256 + k, coutl0 * 16 + i]
                                else:
                                    for j in range(bs_res):
                                        y_data_out[dim, b0 * 8 + j, coutl0 * 16 + i] += x_data_in[dim, b0 * 8 + j, cinl0 * 256 + k] * w_data_in[dim, cinl0 * 256 + k, coutl0 * 16 + i]
                        else:
                            for k in range(cin_res):
                                if (b0 < (bs_loop - 1) or bs_res == 0):
                                    for j in range(8):
                                        y_data_out[dim, b0 * 8 + j, coutl0 * 16 + i] += x_data_in[dim, b0 * 8 + j, cinl0 * 256 + k] * w_data_in[dim, cinl0 * 256 + k, coutl0 * 16 + i]
                                else:
                                    for j in range(bs_res):
                                        y_data_out[dim, b0 * 8 + j, coutl0 * 16 + i] += x_data_in[dim, b0 * 8 + j, cinl0 * 256 + k] * w_data_in[dim, cinl0 * 256 + k, coutl0 * 16 + i]

    return y_data_out

def ann_bf16_matmul(x_data_in, w_data_in, bias_data_in):
    """
    ANN BF16 matmul: bf16 input, bf16 weight, bf16 bias -> bf16 output.

    Input shape: (dim_A, batch, cin) - bf16
    Weight shape: (dim_A, cin, cout) - bf16
    Bias shape: (cout,) - bf16
    Output shape: (dim_A, batch, cout) - bf16
    """
    dim_A, batch, cin = x_data_in.shape
    dim_A2, cin2, cout = w_data_in.shape
    assert (dim_A == dim_A2) and (cin2 == cin), "Input and weight shape mismatch"
    assert cin % 16 == 0, f"For bf16 weight matmul, cin must be a multiple of 16, got {cin}"
    assert cout % 16 == 0, f"For bf16 weight matmul, cout must be a multiple of 16, got {cout}"

    cin_res = cin % 16
    cin_loop = cin // 16 if cin_res == 0 else cin // 16 + 1
    cout_loop = cout // 16
    bs_res = batch % 8
    bs_loop = batch // 8 if (batch % 8 == 0) else (batch // 8 + 1)
    y_data_out = torch.zeros((dim_A, batch, cout), dtype=torch.bfloat16)

    for dim in range(dim_A):
        for b0 in range(bs_loop):
            for coutl0 in range(cout_loop):
                for i in range(16):
                    if (b0 < (bs_loop - 1) or bs_res == 0):
                        for j in range(8):
                            y_data_out[dim, b0 * 8 + j, coutl0 * 16 + i] = bias_data_in[coutl0 * 16 + i]
                    else:
                        for j in range(bs_res):
                            y_data_out[dim, b0 * 8 + j, coutl0 * 16 + i] = bias_data_in[coutl0 * 16 + i]
                for cinl0 in range(cin_loop):
                    for i in range(16):
                        if (cinl0 < (cin_loop - 1) or cin_res == 0):
                            for k in range(16):
                                if (b0 < (bs_loop - 1) or bs_res == 0):
                                    for j in range(8):
                                        y_data_out[dim, b0 * 8 + j, coutl0 * 16 + i] += x_data_in[dim, b0 * 8 + j, cinl0 * 16 + k] * w_data_in[dim, cinl0 * 16 + k, coutl0 * 16 + i]
                                else:
                                    for j in range(bs_res):
                                        y_data_out[dim, b0 * 8 + j, coutl0 * 16 + i] += x_data_in[dim, b0 * 8 + j, cinl0 * 16 + k] * w_data_in[dim, cinl0 * 16 + k, coutl0 * 16 + i]
                        else:
                            for k in range(cin_res):
                                if (b0 < (bs_loop - 1) or bs_res == 0):
                                    for j in range(8):
                                        y_data_out[dim, b0 * 8 + j, coutl0 * 16 + i] += x_data_in[dim, b0 * 8 + j, cinl0 * 16 + k] * w_data_in[dim, cinl0 * 16 + k, coutl0 * 16 + i]
                                else:
                                    for j in range(bs_res):
                                        y_data_out[dim, b0 * 8 + j, coutl0 * 16 + i] += x_data_in[dim, b0 * 8 + j, cinl0 * 16 + k] * w_data_in[dim, cinl0 * 16 + k, coutl0 * 16 + i]

    return y_data_out

def ann_int8_matmul(x_data_in, w_data_in, bias_data_in):
    """
    ANN INT8 matmul: int8 input, int8 weight, int32 bias -> bf16 output.

    Input shape: (dim_A, batch, cin) - int8
    Weight shape: (dim_A, cin, cout) - int8
    Bias shape: (cout,) - int32
    Output shape: (dim_A, batch, cout) - bf16

    Note: Accumulates in int32, then casts to bf16.
    """
    dim_A, batch, cin = x_data_in.shape
    dim_A2, cin2, cout = w_data_in.shape
    assert (dim_A == dim_A2) and (cin2 == cin), "Input and weight shape mismatch"
    assert cin % 32 == 0, f"For int8 weight matmul, cin must be a multiple of 32, got {cin}"
    assert cout % 32 == 0, f"For int8 weight matmul, cout must be a multiple of 32, got {cout}"

    cin_res = cin % 32
    cin_loop = cin // 32 if cin_res == 0 else cin // 32 + 1
    cout_loop = cout // 32
    bs_res = batch % 16
    bs_loop = batch // 16 if (batch % 16 == 0) else (batch // 16 + 1)
    y_data_out = torch.zeros((dim_A, batch, cout), dtype=torch.int32)

    for dim in range(dim_A):
        for b0 in range(bs_loop):
            for coutl0 in range(cout_loop):
                for i in range(32):
                    if (b0 < (bs_loop - 1) or bs_res == 0):
                        for j in range(16):
                            y_data_out[dim, b0 * 16 + j, coutl0 * 32 + i] = bias_data_in[coutl0 * 32 + i]
                    else:
                        for j in range(bs_res):
                            y_data_out[dim, b0 * 16 + j, coutl0 * 32 + i] = bias_data_in[coutl0 * 32 + i]
                for cinl0 in range(cin_loop):
                    for i in range(32):
                        if (cinl0 < (cin_loop - 1) or cin_res == 0):
                            for k in range(32):
                                if (b0 < (bs_loop - 1) or bs_res == 0):
                                    for j in range(16):
                                        y_data_out[dim, b0 * 16 + j, coutl0 * 32 + i] += x_data_in[dim, b0 * 16 + j, cinl0 * 32 + k] * w_data_in[dim, cinl0 * 32 + k, coutl0 * 32 + i]
                                else:
                                    for j in range(bs_res):
                                        y_data_out[dim, b0 * 16 + j, coutl0 * 32 + i] += x_data_in[dim, b0 * 16 + j, cinl0 * 32 + k] * w_data_in[dim, cinl0 * 32 + k, coutl0 * 32 + i]
                        else:
                            for k in range(cin_res):
                                if (b0 < (bs_loop - 1) or bs_res == 0):
                                    for j in range(16):
                                        y_data_out[dim, b0 * 16 + j, coutl0 * 32 + i] += x_data_in[dim, b0 * 16 + j, cinl0 * 32 + k] * w_data_in[dim, cinl0 * 32 + k, coutl0 * 32 + i]
                                else:
                                    for j in range(bs_res):
                                        y_data_out[dim, b0 * 16 + j, coutl0 * 32 + i] += x_data_in[dim, b0 * 16 + j, cinl0 * 32 + k] * w_data_in[dim, cinl0 * 32 + k, coutl0 * 32 + i]

    return y_data_out.to(torch.bfloat16)

def ann_bf16_conv(x_data_in, w_data_in, bias_data_in, stride_h=1, stride_w=1, dilation_h=1, dilation_w=1, padding_top=1, padding_bottom=1, padding_left=1, padding_right=1, padding_value=0):
    """
    ANN BF16 conv: bf16 input, bf16 weight, bf16 bias -> bf16 output.

    Input shape: (bs, xh, xw, cin) - bf16
    Weight shape: (kh, kw, cin, cout) - bf16
    Bias shape: (cout,) - bf16
    Output shape: (bs, yh, yw, cout) - bf16

    Params:
        stride_h/stride_w, dilation_h/dilation_w,
        padding_top/padding_bottom/padding_left/padding_right, padding_value.
    """
    padded_x = F.pad(x_data_in, (0, 0, padding_left, padding_right, padding_top, padding_bottom), value=padding_value)

    bs, xh, xw, cin = x_data_in.shape
    kh, kw, cin2, cout = w_data_in.shape
    assert cin == cin2, "Input channel must match weight channel"
    assert cin % 16 == 0, f"For bf16 conv, cin must be a multiple of 16, got {cin}"
    assert cout % 16 == 0, f"For bf16 conv, cout must be a multiple of 16, got {cout}"

    cin_loop = cin // 16
    cout_loop = cout // 16
    yh = (xh + 1 + 1 - (kh + (dilation_h - 1) * (kh - 1))) // stride_h + 1
    yw = (xw + 1 + 1 - (kw + (dilation_w - 1) * (kw - 1))) // stride_w + 1
    yw_res = yw % 8
    yw_loop = yw // 8 if yw_res == 0 else yw // 8 + 1

    y_data_out = torch.zeros((bs, yh, yw, cout), dtype=torch.bfloat16)

    for b0 in range(bs):
        for yh0 in range(yh):
            for ywl0 in range(yw_loop):
                for coutl0 in range(cout_loop):
                    for i in range(16):
                        if (ywl0 < (yw_loop - 1) or yw_res == 0):
                            for j in range(8):
                                y_data_out[b0, yh0, ywl0 * 8 + j, coutl0 * 16 + i] = bias_data_in[coutl0 * 16 + i]
                        else:
                            for j in range(yw_res):
                                y_data_out[b0, yh0, ywl0 * 8 + j, coutl0 * 16 + i] = bias_data_in[coutl0 * 16 + i]
                    for kh0 in range(kh):
                        for kw0 in range(kw):
                            for cinl0 in range(cin_loop):
                                for i in range(16):
                                    for k in range(16):
                                        if (ywl0 < (yw_loop - 1) or yw_res == 0):
                                            for j in range(8):
                                                y_data_out[b0, yh0, ywl0 * 8 + j, coutl0 * 16 + i] += w_data_in[kh0, kw0,  cinl0 * 16 + k, coutl0 * 16 + i] * padded_x[b0, yh0 * stride_h + kh0 * dilation_h, (ywl0 * 8 + j) * stride_w + kw0 * dilation_w, cinl0 * 16 + k]
                                        else:
                                            for j in range(yw_res):
                                                y_data_out[b0, yh0, ywl0 * 8 + j, coutl0 * 16 + i] += w_data_in[kh0, kw0,  cinl0 * 16 + k, coutl0 * 16 + i] * padded_x[b0, yh0 * stride_h + kh0 * dilation_h, (ywl0 * 8 + j) * stride_w + kw0 * dilation_w, cinl0 * 16 + k]

    return y_data_out

def ann_int8_conv(x_data_in, w_data_in, bias_data_in, stride_h=1, stride_w=1, dilation_h=1, dilation_w=1, padding_top=1, padding_bottom=1, padding_left=1, padding_right=1, padding_value=0):
    """
    ANN INT8 conv: int8 input, int8 weight, int32 bias -> bf16 output.

    Input shape: (bs, xh, xw, cin) - int8
    Weight shape: (kh, kw, cin, cout) - int8
    Bias shape: (cout,) - int32
    Output shape: (bs, yh, yw, cout) - bf16

    Note: Accumulates in int32, then casts to bf16.
    """
    padded_x = F.pad(x_data_in, (0, 0, padding_left, padding_right, padding_top, padding_bottom), value=padding_value)

    bs, xh, xw, cin = x_data_in.shape
    kh, kw, cin2, cout = w_data_in.shape
    assert cin == cin2, "Input channel must match weight channel"

    cin_loop = cin // 32
    cout_loop = cout // 32
    yh = (xh + 1 + 1 - (kh + (dilation_h - 1) * (kh - 1))) // stride_h + 1
    yw = (xw + 1 + 1 - (kw + (dilation_w - 1) * (kw - 1))) // stride_w + 1
    yw_res = yw % 16
    yw_loop = yw // 16 if yw_res == 0 else yw // 16 + 1

    y_data_out = torch.zeros((bs, yh, yw, cout), dtype=torch.int32)

    for b0 in range(bs):
        for yh0 in range(yh):
            for ywl0 in range(yw_loop):
                for coutl0 in range(cout_loop):
                    for i in range(32):
                        if (ywl0 < (yw_loop - 1) or yw_res == 0):
                            for j in range(16):
                                y_data_out[b0, yh0, ywl0 * 16 + j, coutl0 * 32 + i] = bias_data_in[coutl0 * 32 + i]
                        else:
                            for j in range(yw_res):
                                y_data_out[b0, yh0, ywl0 * 16 + j, coutl0 * 32 + i] = bias_data_in[coutl0 * 32 + i]
                    for kh0 in range(kh):
                        for kw0 in range(kw):
                            for cinl0 in range(cin_loop):
                                for i in range(32):
                                    for k in range(32):
                                        if (ywl0 < (yw_loop - 1) or yw_res == 0):
                                            for j in range(16):
                                                y_data_out[b0, yh0, ywl0 * 16 + j, coutl0 * 32 + i] += w_data_in[kh0, kw0,  cinl0 * 32 + k, coutl0 * 32 + i].to(torch.int32) * padded_x[b0, yh0 * stride_h + kh0 * dilation_h, (ywl0 * 16 + j) * stride_w + kw0 * dilation_w, cinl0 * 32 + k].to(torch.int32)
                                        else:
                                            for j in range(yw_res):
                                                y_data_out[b0, yh0, ywl0 * 16 + j, coutl0 * 32 + i] += w_data_in[kh0, kw0,  cinl0 * 32 + k, coutl0 * 32 + i].to(torch.int32) * padded_x[b0, yh0 * stride_h + kh0 * dilation_h, (ywl0 * 16 + j) * stride_w + kw0 * dilation_w, cinl0 * 32 + k].to(torch.int32)

    return y_data_out.to(torch.bfloat16)

def snn_int8_conv(x_data_in, w_data_in, bias_data_in, stride_h=1, stride_w=1, dilation_h=1, dilation_w=1, padding_top=1, padding_bottom=1, padding_left=1, padding_right=1, padding_value=0):
    """
    SNN INT8 conv: spike input, int8 weight, int32 bias -> bf16 output.

    Input shape: (bs, cin, xh, xw) - SPIKE (bool)
    Weight shape: (cout, cin, kh, kw) - int8
    Bias shape: (cout,) - int32
    Output shape: (bs, yh, yw, cout) - bf16

    Note: Accumulates in int32, then casts to bf16.
    """
    padded_x = F.pad(x_data_in, (padding_left, padding_right, padding_top, padding_bottom), value=padding_value)
    
    bs, cin, xh, xw = x_data_in.shape
    cout, cin2, kh, kw = w_data_in.shape
    assert cin == cin2, "Input channel must be equal to weight channel"

    cin_res = cin % 256
    cin_loop = cin // 256 if cin_res == 0 else cin // 256 + 1
    cout_loop = cout // 32
    yh = (xh + padding_top + padding_bottom - (kh + (dilation_h - 1) * (kh - 1))) // stride_h + 1
    yw = (xw + padding_left + padding_right - (kw + (dilation_w - 1) * (kw - 1))) // stride_w + 1
    yw_res = yw % 16
    yw_loop = yw // 16 if yw_res == 0 else yw // 16 + 1

    y_data_out = torch.zeros((bs, yh, yw, cout), dtype=torch.int32)

    for b0 in range(bs):
        for yh0 in range(yh):
            for ywl0 in range(yw_loop):
                for coutl0 in range(cout_loop):
                    for i in range(32):
                        if (ywl0 < (yw_loop - 1) or yw_res == 0):
                            for j in range(16):
                                y_data_out[b0, yh0, ywl0 * 16 + j, coutl0 * 32 + i] = bias_data_in[coutl0 * 32 + i]
                        else:
                            for j in range(yw_res):
                                y_data_out[b0, yh0, ywl0 * 16 + j, coutl0 * 32 + i] = bias_data_in[coutl0 * 32 + i]
                    for kh0 in range(kh):
                        for kw0 in range(kw):
                            for cinl0 in range(cin_loop):
                                for i in range(32):
                                    if (cinl0 < (cin_loop - 1) or cin_res == 0):
                                        for k in range(256):
                                            if (ywl0 < (yw_loop - 1) or yw_res == 0):
                                                for j in range(16):
                                                    y_data_out[b0, yh0, ywl0 * 16 + j, coutl0 * 32 + i] += w_data_in[coutl0 * 32 + i, cinl0 * 256 + k, kh0, kw0].to(torch.int32) * padded_x[b0, cinl0 * 256 + k, yh0 * stride_h + kh0 * dilation_h, (ywl0 * 16 + j) * stride_w + kw0 * dilation_w].to(torch.int32)
                                            else:
                                                for j in range(yw_res):
                                                    y_data_out[b0, yh0, ywl0 * 16 + j, coutl0 * 32 + i] += w_data_in[coutl0 * 32 + i, cinl0 * 256 + k, kh0, kw0].to(torch.int32) * padded_x[b0, cinl0 * 256 + k, yh0 * stride_h + kh0 * dilation_h, (ywl0 * 16 + j) * stride_w + kw0 * dilation_w].to(torch.int32)
                                    else:
                                        for k in range(cin_res):
                                            if (ywl0 < (yw_loop - 1) or yw_res == 0):
                                                for j in range(16):
                                                    y_data_out[b0, yh0, ywl0 * 16 + j, coutl0 * 32 + i] += w_data_in[coutl0 * 32 + i, cinl0 * 256 + k, kh0, kw0].to(torch.int32) * padded_x[b0, cinl0 * 256 + k, yh0 * stride_h + kh0 * dilation_h, (ywl0 * 16 + j) * stride_w + kw0 * dilation_w].to(torch.int32)
                                            else:
                                                for j in range(yw_res):
                                                    y_data_out[b0, yh0, ywl0 * 16 + j, coutl0 * 32 + i] += w_data_in[coutl0 * 32 + i, cinl0 * 256 + k, kh0, kw0].to(torch.int32) * padded_x[b0, cinl0 * 256 + k, yh0 * stride_h + kh0 * dilation_h, (ywl0 * 16 + j) * stride_w + kw0 * dilation_w].to(torch.int32)
                                            
    return y_data_out.to(torch.bfloat16)

def snn_bf16_conv(x_data_in, w_data_in, bias_data_in, stride_h=1, stride_w=1, dilation_h=1, dilation_w=1, padding_top=1, padding_bottom=1, padding_left=1, padding_right=1, padding_value=0):
    """
    SNN BF16 conv: spike input, bf16 weight, bf16 bias -> bf16 output.

    Input shape: (bs, cin, xh, xw) - SPIKE (bool)
    Weight shape: (cout, cin, kh, kw) - bf16
    Bias shape: (cout,) - bf16
    Output shape: (bs, yh, yw, cout) - bf16
    """
    padded_x = F.pad(x_data_in, (padding_left, padding_right, padding_top, padding_bottom), value=padding_value)
    
    bs, cin, xh, xw = x_data_in.shape
    cout, cin2, kh, kw = w_data_in.shape
    assert cin == cin2, "Input channel must be equal to weight channel"

    cin_res = cin % 256
    cin_loop = cin // 256 if cin_res == 0 else cin // 256 + 1
    cout_loop = cout // 16
    yh = (xh + padding_top + padding_bottom - (kh + (dilation_h - 1) * (kh - 1))) // stride_h + 1
    yw = (xw + padding_left + padding_right - (kw + (dilation_w - 1) * (kw - 1))) // stride_w + 1
    yw_res = yw % 8
    yw_loop = yw // 8 if yw_res == 0 else yw // 8 + 1

    y_data_out = torch.zeros((bs, yh, yw, cout), dtype=torch.bfloat16)

    for b0 in range(bs):
        for yh0 in range(yh):
            for ywl0 in range(yw_loop):
                for coutl0 in range(cout_loop):
                    for i in range(16):
                        if (ywl0 < (yw_loop - 1) or yw_res == 0):
                            for j in range(8):
                                y_data_out[b0, yh0, ywl0 * 8 + j, coutl0 * 16 + i] = bias_data_in[coutl0 * 16 + i]
                        else:
                            for j in range(yw_res):
                                y_data_out[b0, yh0, ywl0 * 8 + j, coutl0 * 16 + i] = bias_data_in[coutl0 * 16 + i]
                    for kh0 in range(kh):
                        for kw0 in range(kw):
                            for cinl0 in range(cin_loop):
                                for i in range(16):
                                    if (cinl0 < (cin_loop - 1) or cin_res == 0):
                                        for k in range(256):
                                            if (ywl0 < (yw_loop - 1) or yw_res == 0):
                                                for j in range(8):
                                                    y_data_out[b0, yh0, ywl0 * 8 + j, coutl0 * 16 + i] += w_data_in[coutl0 * 16 + i, cinl0 * 256 + k, kh0, kw0] * padded_x[b0, cinl0 * 256 + k, yh0 * stride_h + kh0 * dilation_h, (ywl0 * 8 + j) * stride_w + kw0 * dilation_w]
                                            else:
                                                for j in range(yw_res):
                                                    y_data_out[b0, yh0, ywl0 * 8 + j, coutl0 * 16 + i] += w_data_in[coutl0 * 16 + i, cinl0 * 256 + k, kh0, kw0] * padded_x[b0, cinl0 * 256 + k, yh0 * stride_h + kh0 * dilation_h, (ywl0 * 8 + j) * stride_w + kw0 * dilation_w]
                                    else:
                                        for k in range(cin_res):
                                            if (ywl0 < (yw_loop - 1) or yw_res == 0):
                                                for j in range(8):
                                                    y_data_out[b0, yh0, ywl0 * 8 + j, coutl0 * 16 + i] += w_data_in[coutl0 * 16 + i, cinl0 * 256 + k, kh0, kw0] * padded_x[b0, cinl0 * 256 + k, yh0 * stride_h + kh0 * dilation_h, (ywl0 * 8 + j) * stride_w + kw0 * dilation_w]
                                            else:
                                                for j in range(yw_res):
                                                    y_data_out[b0, yh0, ywl0 * 8 + j, coutl0 * 16 + i] += w_data_in[coutl0 * 16 + i, cinl0 * 256 + k, kh0, kw0] * padded_x[b0, cinl0 * 256 + k, yh0 * stride_h + kh0 * dilation_h, (ywl0 * 8 + j) * stride_w + kw0 * dilation_w]
                                            
    return y_data_out

def dilate_input(x_data_in: torch.tensor, dilation_h: int, dilation_w: int):
    bs, c, h, w = x_data_in.shape
    h_new = h + (h - 1) * (dilation_h - 1)
    w_new = w + (w - 1) * (dilation_w - 1)
    x_data_out = torch.zeros((bs, c, h_new, w_new), dtype=x_data_in.dtype)
    for i in range(bs):
        for j in range(c):
            for k in range(h_new):
                for l in range(w_new):
                    if (k % dilation_h == 0) and (l % dilation_w == 0):
                        x_data_out[i, j, k, l] = x_data_in[i, j, k // dilation_h, l // dilation_w]
    return x_data_out

def rotate_180(w_data_in: torch.tensor):
    cout, cin, kh, kw = w_data_in.shape
    w_data_out = torch.zeros((cout, cin, kh, kw), dtype=w_data_in.dtype)
    for i in range(cout):
        for j in range(cin):
            for k in range(kh):
                for l in range(kw):
                    w_data_out[i, j, kh - k - 1, kw - l - 1] = w_data_in[i, j, k, l]
    return w_data_out

def write_data_to_file(data_list, test_name, core_x=0, core_y=0):
    os.makedirs(test_name, exist_ok=True)
    os.makedirs(f"{test_name}/golden", exist_ok=True)

    data_list.sort(key=lambda x: x[1])

    with open(f"{test_name}/golden/{core_y}_{core_x}_outputs.txt", "w") as f:
        for data_inst in data_list:
            data, data_addr = data_inst[0], data_inst[1]
            for i in range(len(data)):
                addr_hex = f"{(data_addr+i):04x}"
                data_hex = data[i]
                f.write(f"@{addr_hex} {data_hex}\n")
