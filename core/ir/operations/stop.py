from myhdl import bin, intbv
from typing import Any, Dict, List, Optional, Tuple
from core.ir.operation import Operation
from core.ir.data import Data, DataType, MemBlock
from core.ir.control_op import ControlOp
from core.ir.prims.stop import PrimStop

class Stop(ControlOp):

    def __init__(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, attrs)
        required_attrs = ['jump_addr', 'relative', 'jump']
        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for Stop operation.")
        self.primitive = True  # Marking as a non-primitive operation
    
    
    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        return None


    def para_node(self, inputs: Dict[int, Data], outputs: Dict[int, Data]) -> Optional[Data]:
        """Create a parameter node for Stop."""
        return None
    
    def para_connection(self) -> bool:
        '''
        True: double connection (input and output)
        False: single connection (only input)
        '''
        return False

    def gen_prim(self, inputs: Dict[int, Data], outputs: Dict[int, Data], deps=0b00000000) -> intbv:

        primStop = PrimStop(
            deps=0b00000000, # Placeholder for dependencies
            jump_addr=self.attrs['jump_addr'],
            relative=self.attrs['relative'],
            jump=self.attrs['jump']
        )
        
        return primStop.PIC
    
    def build_prim(self, inputs: Dict[int, Data], outputs: Dict[int, Data], deps=0b00000000):

        primStop = PrimStop(
            deps=0b00000000, # Placeholder for dependencies
            jump_addr=self.attrs['jump_addr'],
            relative=self.attrs['relative'],
            jump=self.attrs['jump']
        )
        
        return primStop
        