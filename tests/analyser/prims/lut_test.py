from core.ir.data import Data, DataType
from core.ir.graph import Graph
from core.ir.hardware_graph import HardwareGraph
from core.compiler.memory_allocator import MemoryAllocator
from core.compiler.operation_scheduler import OperationScheduler

from core.ir.operations.lut import LUT
from core.ir.operations.stop import Stop

from core.simulator.analyser.chip_array import ChipArray as AnalyserArray
from core.simulator.analyser.engine import Engine

from basics.utils import load_config_from_yaml
from tests.utils.soma.utils import make_input_payload_for_lut


def _dtype_to_lut_type(dtype: DataType) -> str:
    if dtype == DataType.INT8:
        return "INT8"
    if dtype == DataType.BF16:
        return "BF16"
    raise ValueError(f"Unsupported dtype for LUT test: {dtype}")


def construct_lut_graph(
    in_dtype: DataType,
    out_dtype: DataType,
    shape: tuple[int, int],
    function: str,
) -> Graph:
    graph = Graph()

    x = make_input_payload_for_lut(in_dtype, shape)
    lut_table = LUT.generate_lut_table(
        x_in_type=_dtype_to_lut_type(in_dtype),
        y_out_type=_dtype_to_lut_type(out_dtype),
        function=function,
    )

    input_data = Data(name="input", dtype=in_dtype, shape=shape, payload=x)
    lut_data = Data(name="lut_table", dtype=out_dtype, shape=lut_table.shape, payload=lut_table)
    output_data = Data(name="output")

    lut_op = LUT("lut_op", {"input_dtype": in_dtype, "output_dtype": out_dtype})

    graph.add_node(input_data)
    graph.add_node(lut_data)
    graph.add_node(output_data)
    graph.add_node(lut_op)

    graph.connect(input_data.name, lut_op.name, dst_port=0)
    graph.connect(lut_data.name, lut_op.name, dst_port=1)
    graph.connect(lut_op.name, output_data.name)

    stop_op = Stop("stop_op", attrs={"jump_addr": 0, "relative": 0, "jump": 1})
    graph.add_node(stop_op)
    graph.connect_control(lut_op.name, stop_op.name)

    graph.infer()
    return graph


def run_one_case(in_dtype: DataType, out_dtype: DataType, shape: tuple[int, int], function: str) -> None:
    graph = construct_lut_graph(in_dtype, out_dtype, shape, function)

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

    print("\nStart running LUT analyser simulation...")
    engine = Engine(chip_array, config)
    engine.run()
    engine.printUtilizations()


if __name__ == "__main__":
    # Use shapes that are exact multiples of per-32B-cell element counts to avoid padding ambiguity.
    combos = [
        (DataType.INT8, DataType.INT8, (3, 64)),
        (DataType.INT8, DataType.BF16, (3, 64)),
        (DataType.BF16, DataType.INT8, (3, 32)),
        (DataType.BF16, DataType.BF16, (3, 32)),
    ]
    functions = ["sigmoid", "tanh", "relu", "gelu"]

    for function in functions:
        for in_dtype, out_dtype, shape in combos:
            print(f"\nRunning LUT analyser case: func={function}, {in_dtype} -> {out_dtype}, shape={shape}")
            run_one_case(in_dtype, out_dtype, shape, function)

    print("\nAll LUT analyser cases finished.")
