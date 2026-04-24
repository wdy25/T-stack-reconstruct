import torch

from core.ir.data import Data, DataType
from core.ir.graph import Graph
from core.ir.hardware_graph import HardwareGraph
from core.compiler.memory_allocator import MemoryAllocator
from core.compiler.operation_scheduler import OperationScheduler

from core.ir.operations.mat_mul import MatMul
from core.ir.operations.stop import Stop

from core.simulator.analyser.chip_array import ChipArray as AnalyserArray
from core.simulator.analyser.engine import Engine

from basics.utils import load_config_from_yaml


def construct_matmul_graph(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> Graph:
    graph = Graph()

    if x.dim() != 2 or w.dim() != 2 or b.dim() != 1:
        raise ValueError(f"Expected x/w/b shapes (2D, 2D, 1D), got {tuple(x.shape)}, {tuple(w.shape)}, {tuple(b.shape)}")
    if x.shape[1] != w.shape[0]:
        raise ValueError(f"Shape mismatch: x is {tuple(x.shape)} but w is {tuple(w.shape)}")
    if w.shape[1] != b.shape[0]:
        raise ValueError(f"Shape mismatch: w is {tuple(w.shape)} but b is {tuple(b.shape)}")

    batch_size, c_in = x.shape
    c_out = w.shape[1]

    dtype_map = {
        torch.int8: DataType.INT8,
        torch.bfloat16: DataType.BF16,
        torch.bool: DataType.SPIKE,
        torch.int32: DataType.INT32,
    }
    try:
        x_dt = dtype_map[x.dtype]
        w_dt = dtype_map[w.dtype]
        b_dt = dtype_map[b.dtype]
    except KeyError as e:
        raise ValueError(f"Unsupported torch dtype in inputs: {e}") from e

    input_x = Data(name="input_x", dtype=x_dt, shape=tuple(x.shape), payload=x)
    weight = Data(name="weight", dtype=w_dt, shape=tuple(w.shape), payload=w)
    bias = Data(name="bias", dtype=b_dt, shape=tuple(b.shape), payload=b)
    output = Data(name="output")

    mm = MatMul(
        name="matmul",
        attrs={
            "in_channels": c_in,
            "out_channels": c_out,
            "batch_size": batch_size,
        },
    )

    graph.add_node(input_x)
    graph.add_node(weight)
    graph.add_node(bias)
    graph.add_node(output)
    graph.add_node(mm)

    graph.connect(input_x.name, mm.name, 0, 0)
    graph.connect(weight.name, mm.name, 0, 1)
    graph.connect(bias.name, mm.name, 0, 2)
    graph.connect(mm.name, output.name, 0, 0)

    stop_op = Stop("stop_op", attrs={"jump_addr": 0, "relative": 0, "jump": 1})
    graph.add_node(stop_op)
    graph.connect_control(mm.name, stop_op.name)

    graph.infer()
    return graph


def run_one_case(name: str, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> None:
    graph = construct_matmul_graph(x, w, b)

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

    config = load_config_from_yaml("core/simulator/configs/basic_config.yaml")
    chip_array = AnalyserArray((1, 1), (1, 1), config=config)
    chip_array.deploy(hardware_graph, op_lists, deps)

    print(f"Start running MatMul analyser simulation: {name}")
    engine = Engine(chip_array, config)
    engine.run()
    engine.printUtilizations()


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 4

    c_in_list = [
        32,
        23,
        16,
        19,
        11,
    ]
    c_out_list = [
        64,
        32,
        48,
        32,
        64,
    ]

    test_vectors = [
        {
            "name": "x int8, w int8, bias int32",
            "x": torch.randint(-4, 4, (batch_size, c_in_list[0]), dtype=torch.int8),
            "w": torch.randint(-4, 4, (c_in_list[0], c_out_list[0]), dtype=torch.int8),
            "b": torch.randint(-64, 64, (c_out_list[0],), dtype=torch.int32),
        },
        {
            "name": "x spike, w int8, bias int32",
            "x": torch.randint(0, 2, (batch_size, c_in_list[1]), dtype=torch.int8).to(torch.bool),
            "w": torch.randint(-4, 4, (c_in_list[1], c_out_list[1]), dtype=torch.int8),
            "b": torch.randint(-64, 64, (c_out_list[1],), dtype=torch.int32),
        },
        {
            "name": "x bf16, w bf16, bias bf16",
            "x": (torch.randn((batch_size, c_in_list[2])) * 0.5).to(torch.bfloat16),
            "w": (torch.randn((c_in_list[2], c_out_list[2])) * 0.5).to(torch.bfloat16),
            "b": (torch.randn((c_out_list[2],)) * 0.5).to(torch.bfloat16),
        },
        {
            "name": "x spike, w bf16, bias bf16",
            "x": torch.randint(0, 2, (batch_size, c_in_list[3]), dtype=torch.int8).to(torch.bool),
            "w": (torch.randn((c_in_list[3], c_out_list[3])) * 0.5).to(torch.bfloat16),
            "b": (torch.randn((c_out_list[3],)) * 0.5).to(torch.bfloat16),
        },
        {
            "name": "x spike, w spike, bias int32",
            "x": torch.randint(0, 2, (batch_size, c_in_list[4]), dtype=torch.int8).to(torch.bool),
            "w": torch.randint(0, 2, (c_in_list[4], c_out_list[4]), dtype=torch.int8).to(torch.bool),
            "b": torch.randint(-64, 64, (c_out_list[4],), dtype=torch.int32),
        },
    ]

    for idx, tv in enumerate(test_vectors, start=1):
        cin = tv["x"].shape[1]
        cout = tv["w"].shape[1]
        print(f"\n===== Case {idx}/5: {tv['name']} (cin={cin}, cout={cout}) =====")
        run_one_case(tv["name"], tv["x"], tv["w"], tv["b"])

    print("\nAll MatMul analyser cases finished.")
