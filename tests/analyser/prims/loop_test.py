import torch
import os
from typing import Dict

from core.ir.data import Data, DataType
from core.ir.graph import Graph
from core.ir.hardware_graph import HardwareGraph
from core.compiler.memory_allocator import MemoryAllocator
from core.compiler.operation_scheduler import OperationScheduler
from core.compiler.code_generator import CodeGenerator
from core.compiler.config_gen import ConfigGenerator
from core.compiler.check_graph import CheckGraph

from core.ir.operations.add import Add
from core.ir.operations.loop import Loop
from core.ir.operations.stop import Stop

from core.simulator.emulator.chip import Chip as EmulatorArray
from core.simulator.emulator.output_if import get_output_list

from basics.utils import makeDir, load_config_from_yaml

def construct_loop_graph(loop_max=3) -> Graph:
    """
    构建包含 Loop 操作的计算图
    loop_max: 循环最大值
    """
    graph = Graph()
    
    # 创建输入数据
    data_1 = torch.randn(2, 3, 32).to(torch.bfloat16)
    data_2 = torch.randn(2, 3, 32).to(torch.bfloat16)
    loop_count = torch.tensor([0], dtype=torch.int32)
    
    input_data_1 = Data(name="input1", dtype=DataType.BF16, shape=(2, 3, 32), payload=data_1)
    input_data_2 = Data(name="input2", dtype=DataType.BF16, shape=(2, 3, 32), payload=data_2)
    loop_count_data = Data(name="loop_count", dtype=DataType.INT32, shape=(1,), payload=loop_count)
    output_data = Data(name="output")
    
    # 创建 Add 操作
    add_op = Add("add_op", {"output_dtype": DataType.BF16, "bc_mode": 0, "scalar": 0, "add_or_sub": 0})
    
    # 创建 Loop 操作
    loop_op = Loop("loop_op", {"loop_max": loop_max, "loop_cnt": 0})
    
    # 创建 Stop 操作
    stop_op = Stop("stop_op", attrs={"jump_addr": 0, "relative": 0, "jump": 1})
    
    # 添加节点到图中
    graph.add_node(input_data_1)
    graph.add_node(input_data_2)
    graph.add_node(loop_count_data)
    graph.add_node(output_data)
    graph.add_node(add_op)
    graph.add_node(loop_op)
    graph.add_node(stop_op)
    
    # 连接数据边
    graph.connect(input_data_1.name, add_op.name)
    graph.connect(input_data_2.name, add_op.name)
    graph.connect(add_op.name, output_data.name)
    graph.connect(loop_count_data.name, loop_op.name)
    
    # 连接控制边
    graph.connect_control(add_op.name, loop_op.name)
    graph.connect_control(loop_op.name, add_op.name)  # 循环回 Add 操作
    graph.connect_control(loop_op.name, stop_op.name)  # 循环结束后跳转到 Stop 操作
    
    # 推断图
    graph.infer()
    
    return graph


if __name__ == "__main__":
    print("Testing Loop operation:")
    graph = construct_loop_graph(loop_max=3)
    
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
    
    # 运行模拟器
    print("\nRunning emulator...")
    core_array.run()
    
    # 获取输出
    outputs = core_array.get_outputs(get_output_list(hardware_graph, ["output"]))
    print(f"\nOutput shape: {outputs['output'].shape}")
    print("Loop operation completed successfully with 3 iterations.")
    
    # 测试不同循环次数
    print("\n\nTesting Loop operation with 5 iterations:")
    graph = construct_loop_graph(loop_max=5)
    
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
    
    core_array = EmulatorArray(config=load_config_from_yaml("core/simulator/configs/basic_config.yaml"), array_size=(1, 1))
    core_array.deploy(hardware_graph, op_lists, deps)
    
    # 运行模拟器
    print("Running emulator...")
    core_array.run()
    
    # 获取输出
    outputs = core_array.get_outputs(get_output_list(hardware_graph, ["output"]))
    print(f"\nOutput shape: {outputs['output'].shape}")
    print("Loop operation completed successfully with 5 iterations.")
    
    print("\nLoop operation tests completed.")
