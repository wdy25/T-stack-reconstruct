from itertools import product
from typing import Dict, Tuple

import torch

from basics.utils import load_config_from_yaml
from core.compiler.memory_allocator import MemoryAllocator
from core.compiler.operation_scheduler import OperationScheduler
from core.ir.data import Data, DataType
from core.ir.graph import Graph
from core.ir.hardware_graph import HardwareGraph
from core.ir.operations.meanpooling import MeanPooling
from core.ir.operations.stop import Stop
from core.simulator.emulator.chip import Chip as EmulatorArray
from core.simulator.emulator.output_if import get_output_list
from tests.utils.utils import relative_error
from tests.utils.soma.utils import make_input_payload_for_meanpooling, compute_expected_torch_meanpooling


def construct_meanpooling_graph(
    output_dtype: DataType,
    scaler_mode: int,
    shape: Tuple[int, int, int, int],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
) -> Tuple[Graph, torch.Tensor]:
    graph = Graph()

    x = make_input_payload_for_meanpooling(shape)
    scaler = 1 if scaler_mode == 1 else 0

    attrs: Dict[str, object] = {
        "kernel_size": kernel_size,
        "stride": stride,
        "scaler": scaler,
        "scaler_mode": scaler_mode,
        "output_dtype": output_dtype,
    }

    y_true = compute_expected_torch_meanpooling(
        x=x,
        kernel_size=kernel_size,
        stride=stride,
        output_dtype=output_dtype,
        scaler_mode=scaler_mode,
        scaler=scaler,
    )

    input_data = Data(name="input", dtype=DataType.BF16, shape=shape, payload=x)
    output_data = Data(name="output")
    meanpool_op = MeanPooling("meanpool_op", attrs=attrs)

    graph.add_node(input_data)
    graph.add_node(output_data)
    graph.add_node(meanpool_op)

    graph.connect(input_data.name, meanpool_op.name, dst_port=0)
    graph.connect(meanpool_op.name, output_data.name, src_port=0)

    stop_op = Stop("stop_op", attrs={"jump_addr": 0, "relative": 0, "jump": 1})
    graph.add_node(stop_op)
    graph.connect_control(meanpool_op.name, stop_op.name)

    graph.infer()
    return graph, y_true


def run_one_case(
    output_dtype: DataType,
    scaler_mode: int,
    shape: Tuple[int, int, int, int],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
) -> None:
    graph, true_output = construct_meanpooling_graph(
        output_dtype=output_dtype,
        scaler_mode=scaler_mode,
        shape=shape,
        kernel_size=kernel_size,
        stride=stride,
    )

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

    assert out.shape == true_output.shape, (
        f"Output shape: {out.shape} mismatch golden shape: {true_output.shape} "
        f"for output_dtype={output_dtype}, scaler_mode={scaler_mode}, "
        f"kernel={kernel_size}, stride={stride}, shape={shape}"
    )

    rel = relative_error(out, true_output)
    assert rel < 1e-2, (
        f"Mismatch for output_dtype={output_dtype}, scaler_mode={scaler_mode}, "
        f"kernel={kernel_size}, stride={stride}, shape={shape}, relative error={rel}"
    )


if __name__ == "__main__":
    spatial_configs = [
        ((2, 7, 7, 16), (5, 5), (2, 2)),
        ((1, 8, 8, 16), (8, 8), (8, 8)),
    ]

    cases = list(
        product(
            (DataType.BF16, DataType.INT8),
            (0, 1),
            spatial_configs,
        )
    )

    for output_dtype, scaler_mode, (shape, kernel_size, stride) in cases:
        print(
            "Running MeanPooling case: "
            f"output_dtype={output_dtype}, scaler_mode={scaler_mode}, "
            f"kernel={kernel_size}, stride={stride}, shape={shape}"
        )
        run_one_case(
            output_dtype=output_dtype,
            scaler_mode=scaler_mode,
            shape=shape,
            kernel_size=kernel_size,
            stride=stride,
        )

    print(f"All MeanPooling cases passed. Total={len(cases)}")
