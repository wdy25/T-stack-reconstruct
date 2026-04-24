from myhdl import bin, intbv
from typing import Any, Dict, List, Optional, Tuple
from core.ir.operation import Operation
from core.ir.data import Data, DataType, MemBlock
from core.ir.control_op import ControlOp
from core.ir.prims.jump import PrimJump

class Jump(ControlOp):

    def __init__(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, attrs)
        # required_attrs = ['jump_addr', 'relative']
        # for attr in required_attrs:
        #     if attr not in self.attrs:
        #         raise ValueError(f"Missing required attribute '{attr}' for Jump operation.")
        self.primitive = True  # Marking as a non-primitive operation
    
    
    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        return None
    
    
    def para_node(self, inputs: Dict[int, Data], outputs: Dict[int, Data]) -> Optional[Data]:
        """Create a parameter node for Jump."""
        return None
    
    def para_connection(self) -> bool:
        '''
        True: double connection (input and output)
        False: single connection (only input)
        '''
        return False
    
    def gen_prim(self, inputs: Dict[int, Data], outputs: Dict[int, Data], jump_addr, deps=0b00000000) -> intbv:
        
        primJump = PrimJump(
            deps=deps, # Placeholder for dependencies
            jump_addr=jump_addr,
            relative=True
        )
        
        return primJump.PIC
        