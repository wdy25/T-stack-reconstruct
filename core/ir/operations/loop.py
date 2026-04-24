from myhdl import bin, intbv
from typing import Any, Dict, List, Optional, Tuple
from core.ir.operation import Operation
from core.ir.data import Data, DataType, MemBlock
from core.ir.control_op import ControlOp
from core.ir.prims.loop import PrimLoop

class Loop(ControlOp):

    def __init__(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, attrs)
        
        required_attrs = ['loop_max', 'loop_cnt']
        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for Jump operation.")
        assert self.attrs['loop_cnt'] <= self.attrs['loop_max'], "loop_cnt must be less than or equal to loop_max."
        assert self.attrs['loop_max'] > 0, "loop_max must be greater than 0."
        assert self.attrs['loop_cnt'] >= 0, "loop_cnt must be non-negative."
        assert self.attrs['loop_max'] < (1<<16), "loop_max must be less than 2^16."
        
        self.primitive = True  # Marking as a non-primitive operation
    
    
    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        return None
    
    
    def para_node(self, inputs: Dict[int, Data], outputs: Dict[int, Data]) -> Optional[Data]:
        """Create a parameter node for Loop."""
        loop_count = intbv(self.attrs['loop_cnt'], min=0, max=(1<<256))
        para_code = MemBlock(length=1, payload=[loop_count])
        para_node = Data(name=f"{self.name}_loop_count", memref=para_code)
        return para_node
    
    def para_connection(self) -> bool:
        '''
        True: double connection (input and output)
        False: single connection (only input)
        '''
        return True
    
    def gen_prim(self, inputs: Dict[int, Data], outputs: Dict[int, Data], jump_addr, deps=0b00000000) -> intbv:
        assert 0 in inputs, "Loop operation requires one input for count."
        assert len(inputs) == 1, "Loop operation requires exactly one input."
        primLoop = PrimLoop(
            deps=deps, # Placeholder for dependencies
            jump_addr=jump_addr,
            config_addr=inputs[0].memref.addr,
            loop_max=self.attrs['loop_max'],
            loop_cnt=self.attrs['loop_cnt'],
            relative=True
        )
        return primLoop.PIC
        