from myhdl import bin, intbv
from typing import Any, Dict, List, Optional, Tuple
from core.ir.operation import Operation
from core.ir.data import Data, DataType, MemBlock
from core.ir.control_op import ControlOp
from core.ir.prims.branch import PrimBranch

class Branch(ControlOp):

    def __init__(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, attrs)
        
        # required_attrs = ['relative']
        # for attr in required_attrs:
        #     if attr not in self.attrs:
        #         raise ValueError(f"Missing required attribute '{attr}' for Jump operation.")
        
        self.primitive = True  # Marking as a non-primitive operation
    
    
    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        assert 0 in inputs, "Jump operation requires one input for condition."
        assert len(inputs) == 1, "Jump operation requires exactly one input."
        condition_data = inputs[0]
        assert condition_data.dtype in (DataType.INT8), "Condition data type must be INT8."
        
        # 检测condition_data只能占一个cell
        # 如果超过1维，则前面的维度之积必须为1
        if len(condition_data.shape) > 1:
            prod = 1
            for dim in condition_data.shape[:-1]:
                prod *= dim
            assert prod == 1, "Condition data for Jump operation must occupy only one memory cell."
        # 最后一个维度不能超过一个cell
        assert condition_data.shape[-1] <= 32, "Condition data for Jump operation must occupy only one memory cell."       
        
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
        assert 0 in inputs, "Jump operation requires one input for condition."
        assert len(inputs) == 1, "Jump operation requires exactly one input."
        primJump = PrimBranch(
            deps=deps, # Placeholder for dependencies
            jump_addr=jump_addr,
            condition_addr=inputs[0].memref.addr,
            relative=True
        )
        
        return primJump.PIC
        