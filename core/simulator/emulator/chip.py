from typing import Dict, List, Tuple, Any, Optional, Set
from myhdl import bin, intbv

from .core import Core
from .router import NOC
from .output_if import OutputIF

from core.ir.hardware_graph import HardwareGraph
from core.ir.graph import NodeId
from core.ir.data import Data, MemBlock
from core.utils.tensor2intbv import intbv2tensor

class Chip():
    def __init__(self, config, array_size) -> None:
        self.config = config
        self.core_array: Dict[Tuple[int, int], Core] = {}
        
        assert type(array_size) == tuple, "Array size should be a tuple"
        assert len(array_size) == 2, "Array size should be a tuple of two elements"
        assert array_size[0] > 0 and array_size[1] > 0, "Array size should be positive"
        
        self.size_y, self.size_x = array_size
        
        self.noc = NOC()
        
        for y in range(self.size_y):
            for x in range(self.size_x):
                self.core_array[(y, x)] = Core(self.config, y, x, self.noc)
    
    
    def __getitem__(self, index) -> Core:
        assert type(index) == tuple, "Index should be a tuple"
        assert len(index) == 2, "Index should be a tuple of two elements"
        return self.core_array[index]

    
    def __setitem__(self, index, core) -> None:
        assert type(index) == tuple, "Index should be a tuple"
        assert len(index) == 2, "Index should be a tuple of two elements"
        self.core_array[index] = core
    
    
    def run(self, from_start=True) -> None:
        checking_finish = 0
        if from_start:
            for core in self.core_array.values():
                core.from_start()
        while True:
            prim_count = 0
            for core in self.core_array.values():
                prim_count += core.run()
            
            if prim_count == 0:
                checking_finish += 1
            else:
                checking_finish = 0
            
            if checking_finish == 2:
                break
        
        success = True
        for core in self.core_array.values():
            if core.stop == False and core.PC < len(core.prims):
                print("Core at position ({}, {}) did not finish".format(core.pos_y, core.pos_x))
                success = False
        
        if success:
            print("All cores finished")
        else:
            print("Some cores did not finish")
    
            
    def deploy(self, hardware_graph: HardwareGraph, op_lists: Dict[Tuple[int,int], List[NodeId]], deps: Dict[Tuple[int,int], List[intbv]]) -> None:
        for core_id, ops in op_lists.items():
            assert core_id in self.core_array, f"Core ID {core_id} not in core array"
            core = self.core_array[core_id]
            prims = []
            for i, nid in enumerate(ops):
                dep = deps[core_id][i]
                node = hardware_graph.node(nid)
                prim = node.build_prim(hardware_graph.input_pairs(nid), hardware_graph.output_pairs(nid), dep)
                prims.append(prim)
            core.prims = prims
            
            # write data
            core_nodes = set(hardware_graph.get_nodes_by_core(core_id))
            for i, nid in enumerate(core_nodes):
                if hardware_graph.kind_of(nid) == "data":
                    data_node = hardware_graph.node(nid)
                    assert isinstance(data_node, Data)
                    if data_node.payload is not None:
                        self.core_array[core_id].memory.writeTensor(data_node.memref.addr, data_node.payload)
                    elif data_node.memref.payload is not None:
                        self.core_array[core_id].memory.writeTensor(data_node.memref.addr, intbv2tensor(data_node.memref.payload))
    
    
    def get_outputs(self, output_if_dict: Dict[NodeId, 'OutputIF']) -> Dict[NodeId, Any]:
        outputs: Dict[NodeId, Any] = {}
        for nid, output_if in output_if_dict.items():
            core_id = output_if.core_id
            core = self.core_array[core_id]
            data = core.memory.readTensor(output_if.addr, output_if.length, output_if.shape, output_if.dtype)
            outputs[nid] = data
        return outputs
        