import os
import sys

import numpy as np
import torch


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.ir.data import Data, DataType
from core.ir.graph import Graph
from core.ir.hardware_graph import HardwareGraph
from core.compiler.memory_allocator import MemoryAllocator
from core.compiler.operation_scheduler import OperationScheduler

from core.ir.operations.nonlinear import Nonlinear
from core.ir.operations.stop import Stop

from core.ir.prims.nonlinear import PrimNonlinear

from core.simulator.analyser.chip_array import ChipArray as AnalyserArray
from core.simulator.analyser.engine import Engine

from basics.utils import load_config_from_yaml
from tests.utils.soma.utils import make_input_payload_for_nonlinear

def construct_nonlinear_graph(func_name: str, shape: tuple[int, int]) -> Graph:
    graph = Graph()

    x = make_input_payload_for_nonlinear(func_name, shape)

    input_data = Data(name="input", dtype=DataType.BF16, shape=shape, payload=x)
    output_data = Data(name="output")

    nonlinear_op = Nonlinear(
        "nonlinear_op",
        {
            "output_dtype": DataType.BF16,
            "function": func_name,
        },
    )

    graph.add_node(input_data)
    graph.add_node(output_data)
    graph.add_node(nonlinear_op)

    graph.connect(input_data.name, nonlinear_op.name)
    graph.connect(nonlinear_op.name, output_data.name)

    stop_op = Stop("stop_op", attrs={"jump_addr": 0, "relative": 0, "jump": 1})
    graph.add_node(stop_op)
    graph.connect_control(nonlinear_op.name, stop_op.name)

    graph.infer()
    return graph


def run_one_case(func_name: str, shape: tuple[int, int]) -> None:
    graph = construct_nonlinear_graph(func_name, shape)

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

    config = load_config_from_yaml("core/simulator/configs/basic_config.yaml")
    chip_array = AnalyserArray((1, 1), (1, 1), config=config)
    chip_array.deploy(hardware_graph, op_lists, deps)

    print("\nStart running Nonlinear analyser simulation...")
    engine = Engine(chip_array, config)
    engine.run()
    engine.printUtilizations()


if __name__ == "__main__":
    supported = PrimNonlinear.get_supported_functions()
    expected_supported = ["sqrt", "sin", "cos", "exp"]
    assert supported == expected_supported, f"get_supported_functions changed: {supported}"

    # Only BF16 -> BF16 for nonlinear
    shape = (3, 32)

    for func_name in supported:
        print(f"\nRunning Nonlinear analyser case: function={func_name}, shape={shape}")
        run_one_case(func_name, shape)

    print("\nAll Nonlinear analyser cases finished.")
