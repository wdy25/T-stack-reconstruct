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

from core.ir.operations.transpose import Transpose
from core.ir.operations.stop import Stop

from core.simulator.analyser.chip_array import ChipArray
from core.simulator.analyser.engine import Engine

from basics.utils import makeDir, load_config_from_yaml


def construct_add_graph() -> Graph:
    graph = Graph()
    
    dim_A, dim_B, dim_C, dim_D = 3,3,3,33
    if (dim_D % 32) != 0:
        dim_D_new = (dim_D // 32) * 32 + 32
    else:
        dim_D_new = dim_D
    x_shape = (dim_A, dim_B, dim_C, dim_D_new)
    x = torch.randint(low=-100,high=100,size=x_shape, dtype=torch.int8)
    transpose_order = 'AD'
    if transpose_order == 'AB':
        permute_order = (1,0,2,3)
    if transpose_order == 'AC':
        permute_order = (2,1,0,3)
    if transpose_order == 'BC':
        permute_order = (0,2,1,3)
    if transpose_order == 'AD':
        permute_order = (3,1,2,0)
    if transpose_order == 'BD':
        permute_order = (0,3,2,1)
    if transpose_order == 'CD':
        permute_order = (0,1,3,2)
    true_output = x.permute(permute_order)
    
    input_data = Data(name="input", dtype=DataType.INT8, shape=x_shape, payload=x)
    output_data = Data(name="output")
    
    transpose_op = Transpose("transpose_op", {"dim_A": dim_A, "dim_B": dim_B, "dim_C": dim_C, "dim_D": dim_D_new, "transpose_order": transpose_order})
    
    graph.add_node(input_data)
    graph.add_node(output_data)
    graph.add_node(transpose_op)
    
    graph.connect(input_data.name, transpose_op.name)
    graph.connect(transpose_op.name, output_data.name)
    
    stop_op = Stop("stop_op", attrs={"jump_addr": 0, "relative": 0, "jump": 1})
    graph.add_node(stop_op)
    graph.connect_control(transpose_op.name, stop_op.name)
    
    graph.infer()
    
    return graph


if __name__ == "__main__":
    graph = construct_add_graph()
    
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
    
    # engine.getTrace(path="temp/add_trace.pb")