import torch

from core.ir.data import Data, DataType
from core.ir.graph import Graph
from core.ir.hardware_graph import HardwareGraph
from core.compiler.memory_allocator import MemoryAllocator
from core.compiler.operation_scheduler import OperationScheduler

from core.ir.operations.convert import Convert
from core.ir.operations.stop import Stop

from core.simulator.analyser.chip_array import ChipArray as AnalyserArray
from core.simulator.analyser.engine import Engine

from basics.utils import load_config_from_yaml
from tests.utils.soma.utils import make_input_payload_for_convert


def construct_convert_graph(in_dtype: DataType, out_dtype: DataType, shape: tuple[int, int]) -> Graph:
    graph = Graph()

    x = make_input_payload_for_convert(in_dtype, shape)

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
    return graph


def run_one_case(in_dtype: DataType, out_dtype: DataType, shape: tuple[int, int]) -> None:
    graph = construct_convert_graph(in_dtype, out_dtype, shape)

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

    print("\nStart running Convert analyser simulation...")
    engine = Engine(chip_array, config)
    engine.run()
    engine.printUtilizations()


if __name__ == "__main__":
    # Use shapes that are exact multiples of per-32B-cell element counts to avoid padding ambiguity.
    cases = [
        (DataType.INT8, DataType.BF16, (3, 64)),
        (DataType.BF16, DataType.INT8, (3, 32)),
        (DataType.INT8, DataType.SPIKE, (3, 256)),
        (DataType.BF16, DataType.SPIKE, (3, 256)),
        (DataType.SPIKE, DataType.INT8, (3, 512)),
        (DataType.SPIKE, DataType.BF16, (3, 512)),
    ]

    for in_dtype, out_dtype, shape in cases:
        print(f"\nRunning Convert case: {in_dtype} -> {out_dtype}, shape={shape}")
        run_one_case(in_dtype, out_dtype, shape)

    print("\nAll Convert analyser cases finished.")
