import torch
from typing import Tuple

from core.ir.data import Data, DataType
from core.ir.graph import Graph
from core.ir.hardware_graph import HardwareGraph
from core.compiler.memory_allocator import MemoryAllocator
from core.compiler.operation_scheduler import OperationScheduler

from core.ir.operations.multiply import Multiply
from core.ir.operations.stop import Stop

from core.simulator.emulator.chip import Chip as EmulatorArray
from core.simulator.emulator.output_if import get_output_list

from basics.utils import load_config_from_yaml


def construct_multiply_graph() -> Tuple[Graph, torch.Tensor]:
    graph = Graph()

    data_1 = torch.randn(2, 3, 32).to(torch.bfloat16)
    data_2 = torch.randn(2, 3, 32).to(torch.bfloat16)
    true_output = (data_1 * data_2).to(torch.bfloat16)

    input_data_1 = Data(name="input1", dtype=DataType.BF16, shape=(2, 3, 32), payload=data_1)
    input_data_2 = Data(name="input2", dtype=DataType.BF16, shape=(2, 3, 32), payload=data_2)
    output_data = Data(name="output")

    multiply_op = Multiply(
        "multiply_op",
        {
            "output_dtype": DataType.BF16,
            "bc_mode": 0,
            "mult_or_div": 0,
            "scalar": 1.0,
        },
    )

    graph.add_node(input_data_1)
    graph.add_node(input_data_2)
    graph.add_node(output_data)
    graph.add_node(multiply_op)

    graph.connect(input_data_1.name, multiply_op.name)
    graph.connect(input_data_2.name, multiply_op.name)
    graph.connect(multiply_op.name, output_data.name)

    stop_op = Stop("stop_op", attrs={"jump_addr": 0, "relative": 0, "jump": 1})
    graph.add_node(stop_op)
    graph.connect_control(multiply_op.name, stop_op.name)

    graph.infer()

    return graph, true_output

if __name__ == "__main__":
    torch.manual_seed(0)

    graph, true_output = construct_multiply_graph()

    hardware_graph = HardwareGraph(graph)
    hardware_graph.gen_memref_for_all_data()
    hardware_graph.set_core_id_for_nodes(hardware_graph.all_nodes(), core_id=(0, 0))
    hardware_graph.gen_communication_ops()

    allocator = MemoryAllocator(hardware_graph)
    hardware_graph.gen_para_nodes()
    allocator.allocate_memory(
        mem_per_core=16384,
        reserved_space=8,
        non_overwritable_patterns=[],
        incremental=False,
    )

    op_sch = OperationScheduler(hardware_graph)
    op_lists = op_sch.build_core_op_lists(try_parallel=True)
    deps = op_sch.build_deps_for_ops(8)

    for core_id, op_list in op_lists.items():
        dep = deps[core_id]
        print(f"\n核心 {core_id} 的调度顺序:")
        for i, nid in enumerate(op_list):
            print(f"  节点 {nid}, 依赖: {bin(dep[i])}")

    core_array = EmulatorArray(
        config=load_config_from_yaml("core/simulator/configs/basic_config.yaml"),
        array_size=(1, 1),
    )
    core_array.deploy(hardware_graph, op_lists, deps)
    core_array.run()

    outputs = core_array.get_outputs(get_output_list(hardware_graph, ["output"]))
    out = outputs["output"]

    error = torch.mean(torch.abs(out - true_output))
    print("absolute error", error)
    assert error == 0, f"Multiply absolute error too large: {error} (expected == 0)"

    print("Multiply operation graph constructed and processed successfully.")