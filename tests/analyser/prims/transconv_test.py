import torch
import os
from typing import Dict
import math

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

from core.simulator.analyser.chip_array import ChipArray
from core.simulator.analyser.engine import Engine

from basics.utils import makeDir, load_config_from_yaml


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
    
    return graph


if __name__ == "__main__":
    graph = construct_transconv_graph()
    
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
    
    chip_array = ChipArray((1, 1), (1, 1), config=load_config_from_yaml("core/simulator/configs/basic_config.yaml"))
    
    chip_array.deploy(hardware_graph, op_lists, deps)
    
    print("\n开始运行仿真...")
    
    engine = Engine(chip_array, load_config_from_yaml("core/simulator/configs/basic_config.yaml"))
    engine.run()
    engine.printUtilizations()