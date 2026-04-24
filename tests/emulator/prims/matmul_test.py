import math
from typing import Tuple

import torch

from core.ir.data import Data, DataType
from core.ir.graph import Graph
from core.ir.hardware_graph import HardwareGraph
from core.compiler.memory_allocator import MemoryAllocator
from core.compiler.operation_scheduler import OperationScheduler
from core.compiler.code_generator import CodeGenerator
from core.compiler.config_gen import ConfigGenerator
from core.compiler.check_graph import CheckGraph

from core.ir.operations.mat_mul import MatMul
from core.ir.operations.stop import Stop

from core.simulator.emulator.chip import Chip as EmulatorArray
from core.simulator.emulator.output_if import get_output_list

from basics.utils import load_config_from_yaml
from tests.utils.utils import relative_error


def _torch_reference_output(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Torch reference for MatMul with bias.

    The primitive's outside-visible output is BF16.

    Notes about dtype handling:
    - For int8/bool inputs, accumulation is int32; then cast to bf16.
    - For bf16 inputs, compute in bf16 directly (no explicit float32 upcast).
    """

    x_dtype = x.dtype
    w_dtype = w.dtype
    b_dtype = b.dtype

    # INT8/BIN spike paths accumulate in INT32, then cast to BF16.
    if b_dtype == torch.int32 and (
        (x_dtype == torch.int8 and w_dtype == torch.int8)
        or (x_dtype == torch.bool and w_dtype == torch.int8)
        or (x_dtype == torch.bool and w_dtype == torch.bool)
    ):
        y_i32 = torch.matmul(x.to(torch.int32), w.to(torch.int32)) + b
        return y_i32.to(torch.bfloat16)

    # BF16 paths: compute directly in BF16 (SPIKE(bool) is treated as 0/1 mask).
    if b_dtype == torch.bfloat16 and w_dtype == torch.bfloat16 and x_dtype in (torch.bfloat16, torch.bool):
        x_bf16 = x if x_dtype == torch.bfloat16 else x.to(torch.bfloat16)
        y_bf16 = torch.matmul(x_bf16, w) + b
        return y_bf16.to(torch.bfloat16)

    raise ValueError(
        "Unsupported matmul dtype combination: "
        f"x={x_dtype}, w={w_dtype}, bias={b_dtype}. "
        "Supported: (int8,int8,int32), (bool,int8,int32), (bf16,bf16,bf16), (bool,bf16,bf16), (bool,bool,int32)."
    )


def construct_matmul_graph(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> Tuple[Graph, torch.Tensor]:
    """Construct a simple 2D MatMul graph: (B, Cin) x (Cin, Cout) + bias -> (B, Cout).

    Note: tensors (x, w, b) are created by the caller so we can reuse this function
    across multiple dtype combinations.
    """

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

    true_output = _torch_reference_output(x, w, b)

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

    return graph, true_output


def run_one_case(idx: int, total: int, tv: dict) -> None:
    print(f"\n===== Case {idx}/{total}: {tv['name']} (cin={tv['c_in']}, cout={tv['c_out']}) =====")
    graph, true_output = construct_matmul_graph(tv["x"], tv["w"], tv["b"])

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
    print("relative_error", rel)
    assert rel < 0.01, f"MatMul relative error too large: {rel} (expected < 0.01)"


# 5 test for different dtype combinations of MatMul
if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 4
    # Handwritten matmul kernels require:
    # - snn int8 weight matmul: cout % 32 == 0
    # - snn bf16 weight matmul: cout % 16 == 0
    # - ann int8 weight matmul: cin % 32 == 0 and cout % 32 == 0
    # - ann bf16 weight matmul: cin % 16 == 0 and cout % 16 == 0
    # Each case uses its own (cin, cout). Keep these aligned with the 5 cases below.
    # NOTE: Some cases have shape constraints (see comments above).
    c_in_list = [
        32,  # case1: ANN int8 requires cin % 32 == 0
        23,  # case2: SNN spike x + int8 w (cin can be arbitrary)
        16,  # case3: ANN bf16 requires cin % 16 == 0
        19,  # case4: SNN spike x + bf16 w (cin can be arbitrary)
        11,  # case5: SNN spike x + spike w (cin can be arbitrary)
    ]
    c_out_list = [
        64,  # case1: ANN int8 requires cout % 32 == 0
        32,  # case2: SNN int8 requires cout % 32 == 0
        48,  # case3: ANN bf16 requires cout % 16 == 0
        32,  # case4: SNN bf16 requires cout % 16 == 0
        64,  # case5: SNN spike weight requires cout % 32 == 0
    ]

    assert len(c_in_list) == 5 and len(c_out_list) == 5

    test_vectors = [
        {
            "name": "x int8, w int8, bias int32, output int32->bf16",
            "c_in": c_in_list[0],
            "c_out": c_out_list[0],
            "x": torch.randint(-4, 4, (batch_size, c_in_list[0]), dtype=torch.int8),
            "w": torch.randint(-4, 4, (c_in_list[0], c_out_list[0]), dtype=torch.int8),
            "b": torch.randint(-64, 64, (c_out_list[0],), dtype=torch.int32),
        },
        {
            "name": "x spike, w int8, bias int32, output int32->bf16",
            "c_in": c_in_list[1],
            "c_out": c_out_list[1],
            "x": torch.randint(0, 2, (batch_size, c_in_list[1]), dtype=torch.int8).to(torch.bool),
            "w": torch.randint(-4, 4, (c_in_list[1], c_out_list[1]), dtype=torch.int8),
            "b": torch.randint(-64, 64, (c_out_list[1],), dtype=torch.int32),
        },
        {
            "name": "x bf16, w bf16, bias bf16, output bf16",
            "c_in": c_in_list[2],
            "c_out": c_out_list[2],
            "x": (torch.randn((batch_size, c_in_list[2])) * 0.5).to(torch.bfloat16),
            "w": (torch.randn((c_in_list[2], c_out_list[2])) * 0.5).to(torch.bfloat16),
            "b": (torch.randn((c_out_list[2],)) * 0.5).to(torch.bfloat16),
        },
        {
            "name": "x spike, w bf16, bias bf16, output bf16",
            "c_in": c_in_list[3],
            "c_out": c_out_list[3],
            "x": torch.randint(0, 2, (batch_size, c_in_list[3]), dtype=torch.int8).to(torch.bool),
            "w": (torch.randn((c_in_list[3], c_out_list[3])) * 0.5).to(torch.bfloat16),
            "b": (torch.randn((c_out_list[3],)) * 0.5).to(torch.bfloat16),
        },
        {
            "name": "x spike, w spike, bias int32, output int32->bf16",
            "c_in": c_in_list[4],
            "c_out": c_out_list[4],
            "x": torch.randint(0, 2, (batch_size, c_in_list[4]), dtype=torch.int8).to(torch.bool),
            "w": torch.randint(0, 2, (c_in_list[4], c_out_list[4]), dtype=torch.int8).to(torch.bool),
            "b": torch.randint(-64, 64, (c_out_list[4],), dtype=torch.int32),
        },
    ]

    total_cases = len(test_vectors)
    for idx, tv in enumerate(test_vectors, start=1):
        run_one_case(idx=idx, total=total_cases, tv=tv)

    print("\nAll 5 MatMul dtype combinations passed.")
