import torch

from core.ir.data import Data, DataType
from core.ir.graph import Graph
from core.ir.hardware_graph import HardwareGraph
from core.compiler.memory_allocator import MemoryAllocator
from core.compiler.operation_scheduler import OperationScheduler

from core.ir.operations.convert import Convert
from core.ir.operations.stop import Stop

from core.simulator.emulator.chip import Chip as EmulatorArray
from core.simulator.emulator.output_if import get_output_list

from basics.utils import load_config_from_yaml
from tests.utils.soma.utils import make_input_payload_for_convert, compute_expected_torch_convert

def construct_convert_graph(in_dtype: DataType, out_dtype: DataType, shape: tuple[int, int]) -> tuple[Graph, torch.Tensor]:
    graph = Graph()

    x = make_input_payload_for_convert(in_dtype, shape)
    y_true = compute_expected_torch_convert(x, in_dtype, out_dtype)

    input_data = Data(name="input", dtype=in_dtype, shape=shape, payload=x)
    output_data = Data(name="output")

    convert_op = Convert("convert_op", {"input_dtype": in_dtype, "output_dtype": out_dtype})

    graph.add_node(input_data)
    graph.add_node(output_data)
    graph.add_node(convert_op)

    graph.connect(input_data.name, convert_op.name)
    graph.connect(convert_op.name, output_data.name)

    stop_op = Stop("stop_op", attrs={"jump_addr": 0, "relative": 0, "jump": 1})
    graph.add_node(stop_op)
    graph.connect_control(convert_op.name, stop_op.name)

    graph.infer()
    return graph, y_true


def run_one_case(in_dtype: DataType, out_dtype: DataType, shape: tuple[int, int]) -> None:
    graph, true_output = construct_convert_graph(in_dtype, out_dtype, shape)

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

    # Compare
    if out_dtype == DataType.SPIKE:
        assert out.dtype == torch.bool
        assert torch.equal(out, true_output), f"Mismatch for {in_dtype} -> {out_dtype}"  # exact
    elif in_dtype == DataType.SPIKE and out_dtype == DataType.BF16:
        # Need bitwise compare to preserve -0.0 semantics.
        assert out.dtype == torch.bfloat16
        assert torch.equal(out.view(torch.int16), true_output.view(torch.int16)), f"Bitwise mismatch for {in_dtype} -> {out_dtype}"
    else:
        assert out.shape == true_output.shape
        diff = torch.mean(torch.abs(out - true_output), dtype=torch.bfloat16).item()
        assert diff == 0.0, f"Numeric mismatch for {in_dtype} -> {out_dtype}, mean abs diff={diff}"


if __name__ == "__main__":
    # Use shapes that are exact multiples of per-32B-cell element counts to avoid padding ambiguity.
    # vector_num_plus_1 = 3 (vectors), input_cells = 2.
    cases = [
        (DataType.INT8, DataType.BF16, (3, 64)),
        (DataType.BF16, DataType.INT8, (3, 32)),
        (DataType.INT8, DataType.SPIKE, (3, 64)),
        (DataType.BF16, DataType.SPIKE, (3, 32)),
        (DataType.SPIKE, DataType.INT8, (3, 512)),
        (DataType.SPIKE, DataType.BF16, (3, 512)),
    ]

    for in_dtype, out_dtype, shape in cases:
        print(f"Running Convert case: {in_dtype} -> {out_dtype}, shape={shape}")
        run_one_case(in_dtype, out_dtype, shape)

    print("All Convert cases passed.")
