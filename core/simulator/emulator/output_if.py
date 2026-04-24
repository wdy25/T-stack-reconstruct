import torch
from typing import List, Dict

from core.ir.hardware_graph import HardwareGraph
from core.ir.graph import NodeId
from core.ir.data import Data, MemBlock, DataType

class OutputIF:
    def __init__(self, addr, length, shape, dtype, core_id) -> None:
        self.addr = addr
        self.length = length
        self.shape = shape
        self.dtype = dtype
        self.core_id = core_id

def torch_dtype(dtype: DataType):
    # INT8 = "int8"
    # BF16 = "bf16"
    # SPIKE = "spike"
    # INT32 = "int32"
    if dtype == DataType.INT8:
        return torch.int8
    if dtype == DataType.BF16:
        return torch.bfloat16
    if dtype == DataType.SPIKE:
        return torch.bool
    if dtype == DataType.INT32:
        return torch.int32
    raise ValueError(f"Unsupported data type: {dtype}")

def get_output_list(hwg: HardwareGraph, nid_list: List[NodeId]) -> Dict[NodeId, OutputIF]:
    output_list: Dict[NodeId, Data] = {}
    for nid in nid_list:
        output_node: Data = hwg.node(nid)
        assert hwg.kind_of(nid) == "data"
        output_list[nid] = OutputIF(
            addr=output_node.memref.addr,
            length=output_node.memref.length,
            shape=output_node.shape,
            dtype=torch_dtype(output_node.dtype),
            core_id=hwg.get_core_id(nid)
        )

    return output_list