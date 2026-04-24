from typing import Dict, List, Tuple, Any, Optional, Set
from myhdl import bin, intbv

from core.ir.graph import Graph, NodeId
from core.ir.hardware_graph import HardwareGraph
from core.ir.data import Data, ViewData, MemBlock
from core.ir.operation import Operation
from core.ir.communication_op import CommOp
from core.ir.control_op import ControlOp
from core.ir.operations import Branch, Jump, Loop, Bar, Stop
from core.ir.prims.stop import PrimStop

class CodeGenerator:
    def __init__(self, hg: HardwareGraph, op_list: Dict[Tuple[int,int], List[NodeId]], deps: Dict[Tuple[int,int], List[intbv]]) -> None:
        self.hg = hg
        self.op_list = op_list
        self.deps = deps
    
    def generate_code(self, auto_stop=True, stop_jump=True, prim_base_addr=0x0) -> Dict[Tuple[int,int], List[intbv]]:
        codes: Dict[Tuple[int,int], List[intbv]] = {}
        for core_id, ops in self.op_list.items():
            codes[core_id] = []
            for i, nid in enumerate(ops):
                node = self.hg.node(nid)
                deps = self.deps[core_id][i]
                assert self.hg.kind_of(nid) in ("operation", "control", "communication")
                if self.hg.kind_of(nid) in ("operation", "communication"):
                    codes[core_id].append(node.gen_prim(self.hg.input_pairs(nid), self.hg.output_pairs(nid), deps))
                elif self.hg.kind_of(nid) == "control":
                    if isinstance(node, Branch):
                        jump_addr = None
                        outputs = self.hg.control_output_pairs(nid)
                        assert outputs is not None and len(outputs) == 2, "Branch control operation must have exactly two outputs."
                        output0 = outputs[0]
                        output1 = outputs[1]
                        assert ops[i+1] == output0.name, "Next operation must be one of the branch targets."
                        for j, jid in enumerate(ops):
                            if jid == output1.name:
                                jump_addr = j - i
                                break
                        if jump_addr is None:
                            raise ValueError("Cannot find jump address for Branch operation.")
                        codes[core_id].append(node.gen_prim(self.hg.input_pairs(nid), self.hg.output_pairs(nid), jump_addr, deps))
                    elif isinstance(node, Jump):
                        jump_addr = None
                        outputs = self.hg.control_output_pairs(nid)
                        assert outputs is not None and len(outputs) == 1, "Jump control operation must have exactly one output."
                        output0 = outputs[0]
                        for j, jid in enumerate(ops):
                            if jid == output0.name:
                                jump_addr = j - i
                                break
                        if jump_addr is None:
                            raise ValueError("Cannot find jump address for Jump operation.")
                        codes[core_id].append(node.gen_prim(self.hg.input_pairs(nid), self.hg.output_pairs(nid), jump_addr, deps))
                    elif isinstance(node, Loop):
                        jump_addr = None
                        outputs = self.hg.control_output_pairs(nid)
                        assert outputs is not None and len(outputs) == 2, "Loop control operation must have exactly two outputs."
                        output0 = outputs[0]
                        output1 = outputs[1]
                        assert ops[i+1] == output0.name, "Next operation must be one of the loop targets."
                        for j, jid in enumerate(ops):
                            if jid == output1.name:
                                jump_addr = j - i
                                break
                        if jump_addr is None:
                            raise ValueError("Cannot find jump address for Loop operation.")
                        codes[core_id].append(node.gen_prim(self.hg.input_pairs(nid), self.hg.output_pairs(nid), jump_addr, deps))
                    elif isinstance(node, Bar):
                        codes[core_id].append(node.gen_prim(self.hg.input_pairs(nid), self.hg.output_pairs(nid), deps))
                    elif isinstance(node, Stop):
                        codes[core_id].append(node.gen_prim(self.hg.input_pairs(nid), self.hg.output_pairs(nid), deps))
                    else:
                        raise NotImplementedError(f"Control operation {type(node)} not supported yet.")

            if auto_stop:
                last_prim = codes[core_id][-1]
                if last_prim[4:0] != 0x0:  # not already a Stop instruction
                    stop_prim = PrimStop(deps=0x0, jump_addr=prim_base_addr, jump=stop_jump, relative=False).PIC
                    codes[core_id].append(stop_prim)
        return codes