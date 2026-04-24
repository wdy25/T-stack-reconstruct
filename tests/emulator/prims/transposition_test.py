import torch

from core.ir.data import Data, DataType
from core.ir.graph import Graph
from core.ir.hardware_graph import HardwareGraph
from core.compiler.memory_allocator import MemoryAllocator
from core.compiler.operation_scheduler import OperationScheduler

from core.ir.operations.transpose import Transpose
from core.ir.operations.stop import Stop

from core.simulator.emulator.chip import Chip as EmulatorArray
from core.simulator.emulator.output_if import get_output_list
from core.utils.get_byte_num import get_torch_dtype_from_DataType

from basics.utils import load_config_from_yaml
from tests.utils.dendrite.utils import make_input_payload_for_transposition, compute_expected_torch_transposition


def construct_transposition_graph(
    in_dtype: DataType,
    shape: tuple[int, int, int, int],
    transpose_order: str,
) -> tuple[Graph, torch.Tensor]:
    graph = Graph()

    x = make_input_payload_for_transposition(in_dtype, shape)
    y_true = compute_expected_torch_transposition(x, transpose_order)

    input_data = Data(name="input", dtype=in_dtype, shape=shape, payload=x)
    output_data = Data(name="output")

    transpose_op = Transpose(
        "transpose_op",
        {
            "dim_A": shape[0],
            "dim_B": shape[1],
            "dim_C": shape[2],
            "dim_D": shape[3],
            "transpose_order": transpose_order,
        },
    )

    graph.add_node(input_data)
    graph.add_node(output_data)
    graph.add_node(transpose_op)

    graph.connect(input_data.name, transpose_op.name)
    graph.connect(transpose_op.name, output_data.name)

    stop_op = Stop("stop_op", attrs={"jump_addr": 0, "relative": 0, "jump": 1})
    graph.add_node(stop_op)
    graph.connect_control(transpose_op.name, stop_op.name)

    graph.infer()
    return graph, y_true


def run_one_case(in_dtype: DataType, shape: tuple[int, int, int, int], transpose_order: str) -> None:
    graph, true_output = construct_transposition_graph(in_dtype, shape, transpose_order)

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

    assert out.dtype == true_output.dtype
    assert out.shape == true_output.shape
    assert torch.equal(out, true_output), f"Mismatch for {in_dtype}, order={transpose_order}, shape={shape}"


if __name__ == "__main__":
    transpose_orders = ["AB", "AC", "AD", "BC", "BD", "CD"]

    # Keep D aligned to 32B-cell element counts to avoid padding ambiguity.
    # INT8: 32 elems / cell, BF16: 16 elems / cell.
    cases: list[tuple[DataType, tuple[int, int, int, int]]] = [
        (DataType.INT8, (2, 3, 4, 32)),
        (DataType.BF16, (2, 3, 4, 16)),
    ]

    for dtype, shape in cases:
        for order in transpose_orders:
            print(f"Running Transposition case: dtype={dtype}, order={order}, shape={shape}")
            run_one_case(dtype, shape, order)

    print("All Transposition cases passed.")