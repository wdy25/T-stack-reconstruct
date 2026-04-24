import numpy as np
import torch

from core.ir.data import Data, DataType
from core.ir.graph import Graph
from core.ir.hardware_graph import HardwareGraph
from core.compiler.memory_allocator import MemoryAllocator
from core.compiler.operation_scheduler import OperationScheduler

from core.ir.operations.nonlinear import Nonlinear
from core.ir.operations.stop import Stop

from core.ir.prims.nonlinear import PrimNonlinear

from core.simulator.emulator.chip import Chip as EmulatorArray
from core.simulator.emulator.output_if import get_output_list

from basics.utils import load_config_from_yaml
from tests.utils.utils import relative_error
from tests.utils.soma.utils import make_input_payload_for_nonlinear, compute_expected_torch_nonlinear

def construct_nonlinear_graph(func_name: str, shape: tuple[int, ...]) -> tuple[Graph, torch.Tensor]:
    graph = Graph()

    x = make_input_payload_for_nonlinear(func_name, shape)
    y_true = compute_expected_torch_nonlinear(x, func_name)

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
    return graph, y_true

def run_one_case(func_name: str, shape: tuple[int, ...]) -> None:
    graph, true_output = construct_nonlinear_graph(func_name, shape)

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

    core_array = EmulatorArray(
        config=load_config_from_yaml("core/simulator/configs/basic_config.yaml"),
        array_size=(1, 1),
    )
    core_array.deploy(hardware_graph, op_lists, deps)
    core_array.run()

    outputs = core_array.get_outputs(get_output_list(hardware_graph, ["output"]))
    out = outputs["output"]

    rel = relative_error(out, true_output)
    print(f"Relative error for Nonlinear({func_name}), shape={shape}: {rel:.2e}")
    assert rel < 1e-2, f"Relative error too large: {rel} (expected < 1e-2)"

    # BF16 exact bitwise compare (covers -0.0 semantics too)
    assert torch.equal(out.view(torch.int16), true_output.view(torch.int16)), (
        f"Bitwise mismatch for Nonlinear({func_name}), shape={shape}"
    )


if __name__ == "__main__":
    supported = PrimNonlinear.get_supported_functions()
    expected_supported = ["sqrt", "sin", "cos", "exp"]
    assert supported == expected_supported, f"get_supported_functions changed: {supported}"

    # Only BF16 -> BF16 for nonlinear
    # Keep 2D shape for simplicity.
    shape = (3, 32)

    for func_name in supported:
        print(f"Running Nonlinear case: function={func_name}, shape={shape}")
        run_one_case(func_name, shape)

    print("All Nonlinear cases passed.")
