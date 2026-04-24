from typing import Dict, List, Tuple, Any, Optional, Set
from myhdl import bin, intbv
from .core import Core

from core.ir.hardware_graph import HardwareGraph
from core.ir.graph import NodeId
from core.ir.data import Data, MemBlock

class ChipArray():
    def __init__(self, chip_array_size, core_array_size, config):
        self.chip_array_size = chip_array_size
        self.core_array_size = core_array_size
        self.config = config
        
        self.cores: Dict[Tuple[int, int], Core] = {}
        for chip_y in range(chip_array_size[0]):
            for chip_x in range(chip_array_size[1]):
                for core_y in range(core_array_size[0]):
                    for core_x in range(core_array_size[1]):
                        chip_pos = (chip_y, chip_x)
                        core_pos = (chip_y * core_array_size[0] + core_y, chip_x * core_array_size[1] + core_x)
                        self.cores[core_pos] = Core(chip_pos, core_pos, config)
    
    
    def __getitem__(self, key):
        return self.cores[key]
    
    
    def __setitem__(self, key, value):
        self.cores[key] = value
        
    
    def deploy(self, hardware_graph: HardwareGraph, op_lists: Dict[Tuple[int,int], List[NodeId]], deps: Dict[Tuple[int,int], List[intbv]]) -> None:
        for core_id, ops in op_lists.items():
            assert core_id in self.cores, f"Core ID {core_id} not in core array"
            core = self.cores[core_id]
            prims = []
            for i, nid in enumerate(ops):
                dep = deps[core_id][i]
                node = hardware_graph.node(nid)
                prim = node.build_prim(hardware_graph.input_pairs(nid), hardware_graph.output_pairs(nid), dep)
                
                # dep 转 dependent_prims
                dependent_prims = []
                for bit in bin(dep)[::-1]:
                    dependent_prims.append(bit == '1')
                prims.append((prim, dependent_prims))
            core.add_prims(prims)