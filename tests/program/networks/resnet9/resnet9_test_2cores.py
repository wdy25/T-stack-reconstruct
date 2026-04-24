import torch
import os
from typing import Dict
import math

from core_components import *
from core.ir.data import Data, DataType, ConcatData, ViewData
from core.ir.graph import Graph
from core_components.hardware_graph import HardwareGraph
from core_components.memory_allocator import MemoryAllocator
from core_components.operation_scheduler import OperationScheduler
from core_components.code_generator import CodeGenerator
from core_components.config_gen import ConfigGenerator

from core.ir.operations.deep_conv import DeepConv
from core.ir.operations.add import Add
from core.ir.operations.maxpooling import MaxPooling
from core.ir.operations.meanpooling import MeanPooling
from core.ir.operations.mat_mul import MatMul
from core.ir.operations.stop import Stop
from core.ir.operations.multiply import Multiply

from ppq_onnx2torch import extractInt8QuantizedOnnx


def quantize_param(layer_info: Dict, layer_key: str, is_fc: bool = False):
    """Return INT8 weight tensor and INT32 bias tensor for given layer key.

    layer_key examples: '/conv1/Conv', '/conv2_1/Conv', '/fc/Gemm'
    Weight in layer_info is assumed in (out_channels, in_channels, k_h, k_w) for conv
    FC weight assumed (out_features, in_features)
    """
    w = torch.tensor(layer_info[layer_key]["weight"], dtype=torch.float32)
    b = torch.tensor(layer_info[layer_key]["bias"], dtype=torch.float32)
    w_scale = layer_info[layer_key]["weight_scale"]
    w_zp = layer_info[layer_key]["weight_zero_point"]
    b_scale = layer_info[layer_key]["bias_scale"]
    b_zp = layer_info[layer_key]["bias_zero_point"]

    w_q = (w / w_scale + w_zp).round().clamp(-128, 127).to(torch.int8)
    b_q = (b / b_scale + b_zp).round().to(torch.int32)

    if is_fc:
        # return (in_features, out_features) for matmul weight (C_in, C_out)
        w_q = w_q.t().contiguous()
    else:
        # Convert to (k_h, k_w, C_in, C_out)
        w_q = w_q.permute(2, 3, 1, 0).contiguous()
    return w_q, b_q


def construct_resnet9_graph(layer_info: Dict, input_tensor: torch.Tensor) -> Graph:
    g = Graph()

    # Input data (convert NCHW -> NHWC expected by conv primitives)
    if input_tensor.dim() != 4:
        raise ValueError("Input tensor must be 4D (N, C, H, W)")
    x_nhwc = input_tensor.permute(0, 2, 3, 1).contiguous()
    x_nhwc = (x_nhwc / layer_info["/conv1/Conv"]["input_scale"]).round().clamp(-128, 127).to(torch.int8)
    
    g.add_node(Data(name="input", shape=x_nhwc.shape, dtype=DataType.INT8, payload=x_nhwc))
    g.add_node(create_reshape_view("input_view", (1, 32, 32, 32)))
    g.connect("input", "input_view")

    # ---- Conv1 + ReLU ----
    conv1_w, conv1_b = quantize_param(layer_info, "/conv1/Conv")
    # weight 拼成 32x3x3x32
    conv1_w_all = torch.zeros((3, 3, 32, 32), dtype=torch.int8)
    conv1_w_all[:, :, 0:conv1_w.shape[2], :] = conv1_w
    g.add_node(Data(name="conv1_weight", shape=(3, 3, 32, 32), dtype=DataType.INT8, payload=conv1_w_all))
    g.add_node(Data(name="conv1_bias", shape=(conv1_b.shape[0],), dtype=DataType.INT32, payload=conv1_b))
    conv1 = DeepConv(name="conv1", attrs={
        "kernel_size": (3, 3), "stride": (1, 1), 
        "padding": (1, 1, 1, 1), "padding_value": 0,
        "dilation": (1, 1), "in_channels": 32, "out_channels": 32
    })
    g.add_node(conv1)
    g.connect("input_view", "conv1", 0, 0)
    g.connect("conv1_weight", "conv1", 0, 1)
    g.connect("conv1_bias", "conv1", 0, 2)
    g.add_node(Data(name="conv1_out", shape=None, dtype=None))
    g.connect("conv1", "conv1_out", 0, 0)
    
    relu1 = MaxPooling(name="relu1", attrs={
        "kernel_size": (1, 1), 
        "stride": (1, 1), 
        "output_dtype": DataType.BF16,
        "scaler_mode": 1,
        "bias_mode": 1,
        "bias_scalar": 0.,
        "scaler": round(math.log2(layer_info["/conv1/Conv"]["bias_scale"]/layer_info["/Add"]["input_1_scale"]))
        })
    g.add_node(relu1)
    g.connect("conv1_out", "relu1", 0, 0)
    g.add_node(Data(name="relu1_out", shape=None, dtype=None))
    g.connect("relu1", "relu1_out", 0, 0)
    
    relu1_scale = MaxPooling(name="relu1_scale", attrs={
        "kernel_size": (1, 1), 
        "stride": (1, 1), 
        "output_dtype": DataType.INT8,
        "scaler_mode": 1,
        "bias_mode": 1,
        "bias_scalar": 0.,
        "scaler": round(math.log2(layer_info["/conv1/Conv"]["bias_scale"]/layer_info["/conv2_1/Conv"]["input_scale"]))
        })
    g.add_node(relu1_scale)
    g.connect("conv1_out", "relu1_scale", 0, 0)
    g.add_node(Data(name="relu1_scale_out", shape=None, dtype=None))
    g.connect("relu1_scale", "relu1_scale_out", 0, 0)

    # # ---- Block 1: conv2_1 -> ReLU -> conv2_2 + residual ----
    conv2_1_w, conv2_1_b = quantize_param(layer_info, "/conv2_1/Conv")
    g.add_node(Data(name="conv2_1_weight", shape=conv2_1_w.shape, dtype=DataType.INT8, payload=conv2_1_w))
    g.add_node(Data(name="conv2_1_bias", shape=(conv2_1_b.shape[0],), dtype=DataType.INT32, payload=conv2_1_b))
    conv2_1 = DeepConv(name="conv2_1", attrs={
        "kernel_size": (3, 3), "stride": (1, 1), "padding": (1, 1, 1, 1),
        "dilation": (1, 1), "in_channels": 32, "out_channels": 32, "padding_value": 0
    })
    g.add_node(conv2_1)
    g.connect("relu1_scale_out", "conv2_1", 0, 0)
    g.connect("conv2_1_weight", "conv2_1", 0, 1)
    g.connect("conv2_1_bias", "conv2_1", 0, 2)
    g.add_node(Data(name="conv2_1_out", shape=None, dtype=None))
    g.connect("conv2_1", "conv2_1_out", 0, 0)
    
    relu2_1 = MaxPooling(name="relu2_1", attrs={
        "kernel_size": (1, 1), 
        "stride": (1, 1), 
        "output_dtype": DataType.INT8,
        "scaler_mode": 1,
        "bias_mode": 1,
        "bias_scalar": 0.,
        "scaler": round(math.log2(layer_info["/conv2_1/Conv"]["bias_scale"]/layer_info["/conv2_2/Conv"]["input_scale"]))
    })
    g.add_node(relu2_1)
    g.connect("conv2_1_out", "relu2_1", 0, 0)
    g.add_node(Data(name="relu2_1_out", shape=None, dtype=None))
    g.connect("relu2_1", "relu2_1_out", 0, 0)

    conv2_2_w, conv2_2_b = quantize_param(layer_info, "/conv2_2/Conv")
    g.add_node(Data(name="conv2_2_weight", shape=conv2_2_w.shape, dtype=DataType.INT8, payload=conv2_2_w))
    g.add_node(Data(name="conv2_2_bias", shape=(conv2_2_b.shape[0],), dtype=DataType.INT32, payload=conv2_2_b))
    conv2_2 = DeepConv(name="conv2_2", attrs={
        "kernel_size": (3, 3), "stride": (1, 1), "padding": (1, 1, 1, 1),
        "dilation": (1, 1), "in_channels": 32, "out_channels": 32, "padding_value": 0
    })
    g.add_node(conv2_2)
    g.connect("relu2_1_out", "conv2_2", 0, 0)
    g.connect("conv2_2_weight", "conv2_2", 0, 1)
    g.connect("conv2_2_bias", "conv2_2", 0, 2)
    g.add_node(Data(name="conv2_2_out", shape=None, dtype=None))
    g.connect("conv2_2", "conv2_2_out", 0, 0)
    
    scale_mul = Multiply(name="scale_mul", attrs={
        "output_dtype": DataType.BF16,
        "bc_mode": 2,
        "mult_or_div": 0,
        "scalar": layer_info["/conv2_2/Conv"]["bias_scale"] / layer_info["/Add"]["input_2_scale"]
    })
    g.add_node(scale_mul)
    g.connect("conv2_2_out", "scale_mul", 0, 0)
    g.add_node(Data(name="scale_mul_out", shape=None, dtype=None))
    g.connect("scale_mul", "scale_mul_out", 0, 0)
    
    add2 = Add(name="add2", attrs={"output_dtype": DataType.BF16, "bc_mode": 0, "add_or_sub": 0})
    g.add_node(add2)
    g.connect("scale_mul_out", "add2", 0, 0)
    g.connect("relu1_out", "add2", 0, 1)
    g.add_node(Data(name="add2_out", shape=None, dtype=None))
    g.connect("add2", "add2_out", 0, 0)
    
    relu2 = MaxPooling(name="relu2", attrs={
        "kernel_size": (1, 1), 
        "stride": (1, 1), 
        "output_dtype": DataType.INT8,
        "scaler_mode": 1,
        "bias_mode": 1,
        "bias_scalar": 0.,
        "scaler": round(math.log2(layer_info["/Add"]["input_1_scale"]/layer_info["/Add"]["output_scale"]))
    })
    g.add_node(relu2)
    g.connect("add2_out", "relu2", 0, 0)
    g.add_node(Data(name="relu2_out", shape=None, dtype=None))
    g.connect("relu2", "relu2_out", 0, 0)

    # ---- Block 2: conv3_1 (stride2) -> ReLU -> conv3_2 + shortcut3 ----
    conv3_1_w, conv3_1_b = quantize_param(layer_info, "/conv3_1/Conv")
    g.add_node(Data(name="conv3_1_weight", shape=conv3_1_w.shape, dtype=DataType.INT8, payload=conv3_1_w))
    g.add_node(Data(name="conv3_1_bias", shape=(conv3_1_b.shape[0],), dtype=DataType.INT32, payload=conv3_1_b))
    conv3_1 = DeepConv(name="conv3_1", attrs={
        "kernel_size": (3, 3), "stride": (2, 2), "padding": (1, 1, 1, 1),
        "dilation": (1, 1), "in_channels": 32, "out_channels": 32, "padding_value": 0
    })
    g.add_node(conv3_1)
    g.connect("relu2_out", "conv3_1", 0, 0)
    g.connect("conv3_1_weight", "conv3_1", 0, 1)
    g.connect("conv3_1_bias", "conv3_1", 0, 2)
    g.add_node(Data(name="conv3_1_out", shape=None, dtype=None))
    g.connect("conv3_1", "conv3_1_out", 0, 0)
    
    relu3_1 = MaxPooling(name="relu3_1", attrs={
        "kernel_size": (1, 1), 
        "stride": (1, 1), 
        "output_dtype": DataType.INT8,
        "scaler_mode": 1,
        "bias_mode": 1,
        "bias_scalar": 0.,
        "scaler": round(math.log2(layer_info["/conv3_1/Conv"]["bias_scale"]/layer_info["/conv3_2/Conv"]["input_scale"]))
    })
    g.add_node(relu3_1)
    g.connect("conv3_1_out", "relu3_1", 0, 0)
    g.add_node(Data(name="relu3_1_out", shape=None, dtype=None))
    g.connect("relu3_1", "relu3_1_out", 0, 0)

    conv3_2_w, conv3_2_b = quantize_param(layer_info, "/conv3_2/Conv")
    g.add_node(Data(name="conv3_2_weight", shape=conv3_2_w.shape, dtype=DataType.INT8, payload=conv3_2_w))
    g.add_node(Data(name="conv3_2_bias", shape=(conv3_2_b.shape[0],), dtype=DataType.INT32, payload=conv3_2_b))
    conv3_2 = DeepConv(name="conv3_2", attrs={
        "kernel_size": (3, 3), "stride": (1, 1), "padding": (1, 1, 1, 1),
        "dilation": (1, 1), "in_channels": 32, "out_channels": 32, "padding_value": 0
    })
    g.add_node(conv3_2)
    g.connect("relu3_1_out", "conv3_2", 0, 0)
    g.connect("conv3_2_weight", "conv3_2", 0, 1)
    g.connect("conv3_2_bias", "conv3_2", 0, 2)
    g.add_node(Data(name="conv3_2_out", shape=None, dtype=None))
    g.connect("conv3_2", "conv3_2_out", 0, 0)
    
    scale_mul3_2 = Multiply(name="scale_mul3_2", attrs={
        "output_dtype": DataType.BF16,
        "bc_mode": 2,
        "mult_or_div": 0,
        "scalar": layer_info["/conv3_2/Conv"]["bias_scale"]/layer_info["/Add_1"]["input_2_scale"]
    })
    g.add_node(scale_mul3_2)
    g.connect("conv3_2_out", "scale_mul3_2", 0, 0)
    g.add_node(Data(name="scale_mul3_2_out", shape=None, dtype=None))
    g.connect("scale_mul3_2", "scale_mul3_2_out", 0, 0)

    shortcut3_w, shortcut3_b = quantize_param(layer_info, "/shortcut3/shortcut3.0/Conv")
    g.add_node(Data(name="shortcut3_weight", shape=shortcut3_w.shape, dtype=DataType.INT8, payload=shortcut3_w))
    g.add_node(Data(name="shortcut3_bias", shape=(shortcut3_b.shape[0],), dtype=DataType.INT32, payload=shortcut3_b))
    shortcut3 = DeepConv(name="shortcut3", attrs={
        "kernel_size": (1, 1), "stride": (2, 2), "padding": (0, 0, 0, 0),
        "dilation": (1, 1), "in_channels": 32, "out_channels": 32, "padding_value": 0
    })
    g.add_node(shortcut3)
    g.connect("relu2_out", "shortcut3", 0, 0)
    g.connect("shortcut3_weight", "shortcut3", 0, 1)
    g.connect("shortcut3_bias", "shortcut3", 0, 2)
    g.add_node(Data(name="shortcut3_out", shape=None, dtype=None))
    g.connect("shortcut3", "shortcut3_out", 0, 0)
    
    # scale_shortcut3 = MaxPooling(name="scale_shortcut3", attrs={
    #     "kernel_size": (1, 1), 
    #     "stride": (1, 1), 
    #     "output_dtype": DataType.BF16,
    #     "scaler_mode": 1,
    #     "bias_mode": 1,
    #     "bias_scalar": float('-inf'),
    #     "scaler": int(round(math.log2(layer_info["/shortcut3/shortcut3.0/Conv"]["bias_scale"]/layer_info["/Add_1"]["input_1_scale"])))
    # })
    scale_shortcut3 = Multiply(name="scale_shortcut3", attrs={
        "output_dtype": DataType.BF16,
        "bc_mode": 2,
        "mult_or_div": 0,
        "scalar": layer_info["/shortcut3/shortcut3.0/Conv"]["bias_scale"]/layer_info["/Add_1"]["input_1_scale"]
    })    
    
    g.add_node(scale_shortcut3)
    g.connect("shortcut3_out", "scale_shortcut3", 0, 0)
    g.add_node(Data(name="scale_shortcut3_out", shape=None, dtype=None))
    g.connect("scale_shortcut3", "scale_shortcut3_out", 0, 0)
    

    add3 = Add(name="add3", attrs={"output_dtype": DataType.BF16, "bc_mode": 0, "add_or_sub": 0})
    g.add_node(add3)
    g.connect("scale_shortcut3_out", "add3", 0, 0)
    g.connect("scale_mul3_2_out", "add3", 0, 1)
    g.add_node(Data(name="add3_out", shape=None, dtype=None))
    g.connect("add3", "add3_out", 0, 0)
    
    relu3 = MaxPooling(name="relu3", attrs={
        "kernel_size": (1, 1), 
        "stride": (1, 1), 
        "output_dtype": DataType.INT8,
        "scaler_mode": 1,
        "bias_mode": 1,
        "bias_scalar": 0.,
        "scaler": round(math.log2(layer_info["/Add_1"]["input_1_scale"]/layer_info["/Add_1"]["output_scale"]))
    })
    g.add_node(relu3)
    g.connect("add3_out", "relu3", 0, 0)
    g.add_node(Data(name="relu3_out", shape=None, dtype=None))
    g.connect("relu3", "relu3_out", 0, 0)

    # ---- Block 3: conv4_1 (stride2) -> ReLU -> conv4_2 + shortcut4 ----
    conv4_1_w, conv4_1_b = quantize_param(layer_info, "/conv4_1/Conv")
    g.add_node(Data(name="conv4_1_weight", shape=conv4_1_w.shape, dtype=DataType.INT8, payload=conv4_1_w))
    g.add_node(Data(name="conv4_1_bias", shape=(conv4_1_b.shape[0],), dtype=DataType.INT32, payload=conv4_1_b))
    conv4_1 = DeepConv(name="conv4_1", attrs={
        "kernel_size": (3, 3), "stride": (2, 2), "padding": (1, 1, 1, 1),
        "dilation": (1, 1), "in_channels": 32, "out_channels": 64, "padding_value": 0
    })
    g.add_node(conv4_1)
    g.connect("relu3_out", "conv4_1", 0, 0)
    g.connect("conv4_1_weight", "conv4_1", 0, 1)
    g.connect("conv4_1_bias", "conv4_1", 0, 2)
    g.add_node(Data(name="conv4_1_out", shape=None, dtype=None))
    g.connect("conv4_1", "conv4_1_out", 0, 0)
    
    relu4_1 = MaxPooling(name="relu4_1", attrs={
        "kernel_size": (1, 1), 
        "stride": (1, 1), 
        "output_dtype": DataType.INT8,
        "scaler_mode": 1,
        "bias_mode": 1,
        "bias_scalar": 0.,
        "scaler": round(math.log2(layer_info["/conv4_1/Conv"]["bias_scale"]/layer_info["/conv4_2/Conv"]["input_scale"]))
    })
    g.add_node(relu4_1)
    g.connect("conv4_1_out", "relu4_1", 0, 0)
    g.add_node(Data(name="relu4_1_out", shape=None, dtype=None))
    g.connect("relu4_1", "relu4_1_out", 0, 0)

    conv4_2_w, conv4_2_b = quantize_param(layer_info, "/conv4_2/Conv")
    g.add_node(Data(name="conv4_2_weight", shape=conv4_2_w.shape, dtype=DataType.INT8, payload=conv4_2_w))
    g.add_node(Data(name="conv4_2_bias", shape=(conv4_2_b.shape[0],), dtype=DataType.INT32, payload=conv4_2_b))
    conv4_2 = DeepConv(name="conv4_2", attrs={
        "kernel_size": (3, 3), "stride": (1, 1), "padding": (1, 1, 1, 1),
        "dilation": (1, 1), "in_channels": 64, "out_channels": 64, "padding_value": 0
    })
    g.add_node(conv4_2)
    g.connect("relu4_1_out", "conv4_2", 0, 0)
    g.connect("conv4_2_weight", "conv4_2", 0, 1)
    g.connect("conv4_2_bias", "conv4_2", 0, 2)
    g.add_node(Data(name="conv4_2_out", shape=None, dtype=None))
    g.connect("conv4_2", "conv4_2_out", 0, 0)
    
    scale_mul4_2 = Multiply(name="scale_mul4_2", attrs={
        "output_dtype": DataType.BF16,
        "bc_mode": 2,
        "mult_or_div": 0,
        "scalar": layer_info["/conv4_2/Conv"]["bias_scale"]/layer_info["/Add_2"]["input_2_scale"]
    })
    g.add_node(scale_mul4_2)
    g.connect("conv4_2_out", "scale_mul4_2", 0, 0)
    g.add_node(Data(name="scale_mul4_2_out", shape=None, dtype=None))
    g.connect("scale_mul4_2", "scale_mul4_2_out", 0, 0)

    shortcut4_w, shortcut4_b = quantize_param(layer_info, "/shortcut4/shortcut4.0/Conv")
    g.add_node(Data(name="shortcut4_weight", shape=shortcut4_w.shape, dtype=DataType.INT8, payload=shortcut4_w))
    g.add_node(Data(name="shortcut4_bias", shape=(shortcut4_b.shape[0],), dtype=DataType.INT32, payload=shortcut4_b))
    shortcut4 = DeepConv(name="shortcut4", attrs={
        "kernel_size": (1, 1), "stride": (2, 2), "padding": (0, 0, 0, 0),
        "dilation": (1, 1), "in_channels": 32, "out_channels": 64, "padding_value": 0
    })
    g.add_node(shortcut4)
    g.connect("relu3_out", "shortcut4", 0, 0)
    g.connect("shortcut4_weight", "shortcut4", 0, 1)
    g.connect("shortcut4_bias", "shortcut4", 0, 2)
    g.add_node(Data(name="shortcut4_out", shape=None, dtype=None))
    g.connect("shortcut4", "shortcut4_out", 0, 0)
    
    scale_shortcut4 = Multiply(name="scale_shortcut4", attrs={
        "output_dtype": DataType.BF16,
        "bc_mode": 2,
        "mult_or_div": 0,
        "scalar": layer_info["/shortcut4/shortcut4.0/Conv"]["bias_scale"]/layer_info["/Add_2"]["input_1_scale"]
    })    
    
    g.add_node(scale_shortcut4)
    g.connect("shortcut4_out", "scale_shortcut4", 0, 0)
    g.add_node(Data(name="scale_shortcut4_out", shape=None, dtype=None))
    g.connect("scale_shortcut4", "scale_shortcut4_out", 0, 0)
    
    

    add4 = Add(name="add4", attrs={"output_dtype": DataType.BF16, "bc_mode": 0, "add_or_sub": 0})
    g.add_node(add4)
    g.connect("scale_mul4_2_out", "add4", 0, 0)
    g.connect("scale_shortcut4_out", "add4", 0, 1)
    g.add_node(Data(name="add4_out", shape=None, dtype=None))
    g.connect("add4", "add4_out", 0, 0)
    relu4 = MaxPooling(name="relu4", attrs={
        "kernel_size": (1, 1), 
        "stride": (1, 1), 
        "output_dtype": DataType.BF16,
        "scaler_mode": 1,
        "bias_mode": 1,
        "bias_scalar": 0.,
        "scaler": round(math.log2(layer_info["/Add_2"]["input_1_scale"]/layer_info["/Add_2"]["output_scale"]))
    })
    g.add_node(relu4)
    g.connect("add4_out", "relu4", 0, 0)
    g.add_node(Data(name="relu4_out", shape=None, dtype=None))
    g.connect("relu4", "relu4_out", 0, 0)

    # ---- AvgPool (kernel 8 stride 8) ----
    # Create identity weight for avg pool conv (float; will not quantize strictly)
    avg_pool = MeanPooling(name="avg_pool", attrs={
        "kernel_size": (8, 8),
        "stride": (8, 8),
        "output_dtype": DataType.INT8,
        "scaler_mode": 1,
        "scaler": round(math.log2(layer_info["/Add_2"]["output_scale"]/layer_info["/fc/Gemm"]["input_scale"]/64)),
    })
    g.add_node(avg_pool)
    g.connect("relu4_out", "avg_pool", 0, 0)
    g.add_node(Data(name="avg_out", shape=None, dtype=None))
    g.connect("avg_pool", "avg_out", 0, 0)

    # ---- FC (MatMul) ----
    fc_w, fc_b = quantize_param(layer_info, "/fc/Gemm", is_fc=True)
    fc_w_all = torch.zeros((64, 32), dtype=torch.int8)
    fc_w_all[:, 0:fc_w.shape[1]] = fc_w
    fc_b_all = torch.zeros((32,), dtype=torch.int32)
    fc_b_all[0:fc_b.shape[0]] = fc_b
    # reshape avg_out (N,1,1,64) -> (N,64)
    view_fc = ViewData(name="avg_view", view_type="reshape", target_shape=[1, 64])
    g.add_node(view_fc)
    g.connect("avg_out", "avg_view", 0, 0)
    g.add_node(Data(name="fc_weight", shape=(64, 32), dtype=DataType.INT8, payload=fc_w_all))
    g.add_node(Data(name="fc_bias", shape=(32,), dtype=DataType.INT32, payload=fc_b_all))
    fc = MatMul(name="fc", attrs={
        "in_channels": 64, 
        "out_channels": 32, 
        "batch_size": 1
    })
    g.add_node(fc)
    g.connect("avg_view", "fc", 0, 0)
    g.connect("fc_weight", "fc", 0, 1)
    g.connect("fc_bias", "fc", 0, 2)
    g.add_node(Data(name="fc_out", shape=None, dtype=None))
    g.connect("fc", "fc_out", 0, 0)
    
    out_view = ViewData(name="out_view", view_type="reshape", target_shape=[1, 1, 1, 32])
    g.add_node(out_view)
    g.connect("fc_out", "out_view", 0, 0)
    
    scale_out = MaxPooling(name="scale_out", attrs={
        "kernel_size": (1, 1), 
        "stride": (1, 1), 
        "output_dtype": DataType.INT8,
        "scaler_mode": 1,
        "bias_mode": 1,
        "bias_scalar": float('-inf'),
        "scaler": round(math.log2(layer_info["/fc/Gemm"]["bias_scale"]/layer_info["/fc/Gemm"]["output_scale"]))
    })
    g.add_node(scale_out)
    g.connect("out_view", "scale_out", 0, 0)
    g.add_node(Data(name="output", shape=None, dtype=None))
    g.connect("scale_out", "output", 0, 0)

    g.infer()
    g.to_prim()
    return g


def main():
    # Load ONNX quant info
    layer_info = extractInt8QuantizedOnnx("test_gen/network/resnet9/MQuantized.onnx")
    # Load sample input (already tensor NCHW)
    image = torch.load("test_gen/network/resnet9/input_data/truck.tensor")
    g = construct_resnet9_graph(layer_info, image)

    hwg = HardwareGraph(g)
    gs = hwg.split([("relu2", "relu2_out")])
    hwg.set_core_id_for_nodes(gs[0], core_id=(0, 0))
    hwg.set_core_id_for_nodes(gs[1], core_id=(0, 1))
    # hwg.gen_memref_for_all_data()
    hwg.gen_communication_ops()
    hwg.gen_memref_for_all_data()
    allocator = MemoryAllocator(hwg)
    hwg.gen_para_nodes()
    allocator.allocate_memory(mem_per_core=16384, reserved_space=256, non_overwritable_patterns=['.*weight', '.*bias', '.*para.*', 'input'], incremental=True)

    hwg.visualize("testcases/resnet9_2_cores/resnet9_hardware_graph", format="pdf", vertical=True)
    allocator.visualize_lifecycle("testcases/resnet9_2_cores/resnet9_memory_lifecycle")

    op_sch = OperationScheduler(hwg)
    op_lists = op_sch.build_core_op_lists(try_parallel=True)
    deps = op_sch.build_deps_for_ops(8)
    for core_id, op_list in op_lists.items():
        dep = deps[core_id]
        print(f"\n核心 {core_id} 的调度顺序:")
        for i, nid in enumerate(op_list):
            print(f"  节点 {nid}, 依赖: {bin(dep[i])}")

    CheckGraph().check(hwg)
    code_gen = CodeGenerator(hwg, op_lists, deps)
    codes = code_gen.generate_code()
    for core_id, code in codes.items():
        print(f"\n核心 {core_id} 的指令:")
        for i, instr in enumerate(code):
            print(f"  指令 {i}: {bin(instr)}")

    config_gen = ConfigGenerator(hwg, codes, output_dir="testcases/resnet9_2_cores/")
    config_gen.generate_all_configs()
    print("ResNet9 test graph 完成。")


if __name__ == "__main__":
    main()
