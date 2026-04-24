import torch
import os
from typing import Dict
import math
import torch.nn.functional as F
from core.ir.data import Data, DataType, ConcatData, ViewData, create_reshape_view
from core.ir.graph import Graph
from core.ir.hardware_graph import HardwareGraph
from core.compiler.memory_allocator import MemoryAllocator
from core.compiler.operation_scheduler import OperationScheduler
from core.compiler.code_generator import CodeGenerator
from core.compiler.config_gen import ConfigGenerator
from core.compiler.check_graph import CheckGraph

from core.ir.operations.trans_conv import TransConv
from core.ir.operations.stop import Stop

from core.simulator.emulator.chip import Chip as EmulatorArray
from core.simulator.emulator.output_if import get_output_list

from basics.utils import makeDir, load_config_from_yaml

import torch

def dilate_input(x_data_in: torch.tensor, dilation_h: int, dilation_w: int):
    """
    对输入特征图进行膨胀操作（在元素间插入零值）
    :param x_data_in: 输入张量，形状为 (batch_size, channels, height, width)
    :param dilation_h: 高度方向的膨胀率
    :param dilation_w: 宽度方向的膨胀率
    :return: 膨胀后的张量
    """
    bs, c, h, w = x_data_in.shape
    # 计算膨胀后的尺寸
    h_new = h + (h - 1) * (dilation_h - 1)
    w_new = w + (w - 1) * (dilation_w - 1)
    
    # 初始化输出张量为全零
    x_data_out = torch.zeros((bs, c, h_new, w_new), dtype=x_data_in.dtype)
    
    # 遍历每个维度，将原始值填充到膨胀后的对应位置
    for i in range(bs):
        for j in range(c):
            for k in range(h_new):
                for l in range(w_new):
                    # 仅在膨胀率整数倍的位置填充原始值
                    if (k % dilation_h == 0) and (l % dilation_w == 0):
                        x_data_out[i, j, k, l] = x_data_in[i, j, k // dilation_h, l // dilation_w]
    return x_data_out

def rotate_180(w_data_in: torch.tensor):
    """
    将卷积核张量旋转180度
    :param w_data_in: 卷积核张量，形状为 (out_channels, in_channels, kernel_h, kernel_w)
    :return: 旋转180度后的卷积核
    """
    cout, cin, kh, kw = w_data_in.shape
    
    # 初始化输出张量为全零
    w_data_out = torch.zeros((cout, cin, kh, kw), dtype=w_data_in.dtype)
    
    # 遍历每个维度，将元素旋转180度填充
    for i in range(cout):
        for j in range(cin):
            for k in range(kh):
                for l in range(kw):
                    # 180度旋转的坐标映射：(k,l) → (kh-k-1, kw-l-1)
                    w_data_out[i, j, kh - k - 1, kw - l - 1] = w_data_in[i, j, k, l]
    return w_data_out

def handwritten_bf16_conv(padded_x: torch.Tensor,w: torch.Tensor,b: torch.Tensor,y: torch.Tensor, sh, sw, dh, dw):
    assert padded_x.dtype == torch.bfloat16
    assert w.dtype == torch.bfloat16
    assert b.dtype == torch.bfloat16
    assert y.dtype == torch.bfloat16

    batch, cin, xh, xw = padded_x.shape
    cout, _, kh, kw = w.shape
    _, _, yh, yw = y.shape

    assert (cin % 16) == 0, "cin % 16 != 0"
    assert (cout % 16) == 0, "cout % 16 != 0"

    cinloop = int(cin / 16)
    coutloop = int(cout / 16)
    ywres = yw % 8
    if (ywres == 0):
        ywloop = yw // 8
    else:
        ywloop = yw // 8 + 1

    for b0 in range(batch):
        for yh0 in range(yh):
            for ywl0 in range(ywloop):
                for coutloop0 in range(coutloop):
                    for i in range(16):
                        if (ywl0 < (ywloop - 1) or ywres == 0):
                            for j in range(8):
                                y[b0, coutloop0*16+i, yh0, ywl0*8+j] = b[coutloop0*16+i]
                        else:
                            for j in range(ywres):
                                y[b0, coutloop0*16+i, yh0, ywl0*8+j] = b[coutloop0*16+i]
                    for kh0 in range(kh):
                        for kw0 in range(kw):
                            for cinloop0 in range(cinloop):
                                for i in range(16):
                                    for k in range(16):
                                        if (ywl0 < (ywloop - 1) or ywres == 0):
                                            for j in range(8):
                                                y[b0, coutloop0*16+i, yh0, ywl0*8+j] += padded_x[b0, cinloop0*16+k, yh0*dh+kh0*sh, (ywl0*8+j)*dw+kw0*sw] * w[coutloop0*16+i, cinloop0*16+k, kh0, kw0]
                                        else:
                                            for j in range(ywres):
                                                y[b0, coutloop0*16+i, yh0, ywl0*8+j] += padded_x[b0, cinloop0*16+k, yh0*dh+kh0*sh, (ywl0*8+j)*dw+kw0*sw] * w[coutloop0*16+i, cinloop0*16+k, kh0, kw0]

    return y

def construct_transconv_graph() -> Graph:
    graph = Graph()
    params = ['BF16',8,8,16,16,3,3,1,1,1,1,0,0,0,0,1,0,0]
    conv_type = params[0]
    x_in_w, x_in_h = params[1], params[2]
    c_in, c_out = params[3],params[4]
    k_h, k_w = params[5],params[6]
    dilation_h, dilation_w = params[7],params[8]
    xhjump, xwjump = params[9],params[10]
    padding_top, padding_bottom, padding_left, padding_right = params[11],params[12],params[13],params[14]
    padding_top_new = dilation_h * (k_h - 1) - padding_top
    padding_bottom_new = dilation_h * (k_h - 1) - padding_bottom
    padding_left_new = dilation_w * (k_w - 1) - padding_left
    padding_right_new = dilation_w * (k_w - 1) - padding_right
    padding_value = params[17]
    bs = params[15]
    isSNN = params[16]
    x_shape = (bs,c_in,x_in_h,x_in_w)
    w_shape = (c_out,c_in,k_h,k_w)
    b_shape = (c_out,)
    y_out_w = x_in_w + (x_in_w - 1) * (xwjump - 1) + padding_left_new + padding_right_new - (k_w + (dilation_w - 1) * (k_w - 1)) + 1
    y_out_h = x_in_h + (x_in_h - 1) * (xhjump - 1) + padding_top_new + padding_bottom_new - (k_h + (dilation_h - 1) * (k_h - 1)) + 1
    
    y_out = torch.zeros((bs,c_out,y_out_h,y_out_w), dtype=torch.bfloat16)
    x_in = torch.rand(x_shape, dtype=torch.bfloat16)
    w_in = torch.rand(w_shape, dtype=torch.bfloat16)
    b_in = torch.rand(b_shape,dtype=torch.bfloat16)

    x_in_compute = dilate_input(x_in, xhjump, xwjump)
    w_in_compute = rotate_180(w_in)
    padded_x_in = F.pad(x_in_compute,(padding_left_new,padding_right_new,padding_top_new,padding_bottom_new),value=padding_value)
    true_output = handwritten_bf16_conv(padded_x_in,w_in_compute,b_in,y_out,1,1,dilation_h,dilation_w)
    input = Data(name="TransConv_input", dtype=DataType.BF16, shape=(bs, x_in_h, x_in_w, c_in), payload=x_in.permute(0,2,3,1))
    weight = Data(name="TransConv_weight", dtype=DataType.BF16, shape=(k_h, k_w, c_in, c_out), payload=w_in.permute(2,3,1,0))
    bias = Data(name="TransConv_bias", dtype=DataType.BF16, shape=b_shape, payload=b_in)
    output_data = Data(name="TransConv_output", dtype=None, shape=None)
    
    transconv_op = TransConv(name="transconv_op", attrs={
        "kernel_size": (k_h, k_w), "stride": (1, 1), 
        "padding": (padding_top_new, padding_bottom_new, padding_left_new, padding_right_new), "padding_value": padding_value,
        "dilation": (dilation_h, dilation_w), "input_dilation": (xhjump, xwjump), "in_channels": c_in, "out_channels": c_out
    })
    
    graph.add_node(input)
    graph.add_node(weight)
    graph.add_node(bias)
    graph.add_node(output_data)
    graph.add_node(transconv_op)
    
    graph.connect(input.name, transconv_op.name)
    graph.connect(weight.name, transconv_op.name)
    graph.connect(bias.name, transconv_op.name)
    graph.connect(transconv_op.name, output_data.name)
    
    stop_op = Stop("stop_op", attrs={"jump_addr": 0, "relative": 0, "jump": 1})
    graph.add_node(stop_op)
    graph.connect_control(transconv_op.name, stop_op.name)
    
    graph.infer()
    
    return graph, true_output


if __name__ == "__main__":
    graph, true_output = construct_transconv_graph()
    
    hardware_graph = HardwareGraph(graph)
    hardware_graph.gen_memref_for_all_data()
    hardware_graph.set_core_id_for_nodes(hardware_graph.all_nodes(), core_id=(0, 0))
    hardware_graph.gen_communication_ops()
    allocator = MemoryAllocator(hardware_graph)
    hardware_graph.gen_para_nodes()
    allocator.allocate_memory(mem_per_core=16384, reserved_space=8, non_overwritable_patterns=[], incremental=False)
    op_sch = OperationScheduler(hardware_graph)
    op_lists = op_sch.build_core_op_lists(try_parallel=True)
    deps = op_sch.build_deps_for_ops(8)
    for core_id, op_list in op_lists.items():
        dep = deps[core_id]
        print(f"\n核心 {core_id} 的调度顺序:")
        for i, nid in enumerate(op_list):
            print(f"  节点 {nid}, 依赖: {bin(dep[i])}")
            
    core_array = EmulatorArray(config=load_config_from_yaml("core/simulator/configs/basic_config.yaml"), array_size=(1, 1))
    core_array.deploy(hardware_graph, op_lists, deps)
    
    core_array.run()
    
    outputs = core_array.get_outputs(get_output_list(hardware_graph, ["TransConv_output"]))
    print(torch.sum(torch.abs(outputs["TransConv_output"] - true_output.permute(0,2,3,1))))
    
    print("TransConv operation graph constructed and processed successfully.")