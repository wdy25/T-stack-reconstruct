from myhdl import bin, intbv
from typing import Any, Dict, List, Optional, Tuple
from core.ir.operation import Operation
from core.ir.data import Data, DataType, MemBlock, ViewData
from core.ir.prims.split import PrimSplit
import math

class Split(Operation):
    """Split tensor into two outputs with length and count controls.

    Ports:
        inputs:
            - 0: input tensor (2D (vector_num, vector_len), BF16/INT8 supported)
        outputs:
            - 0: first output tensor (2D (output1_num, output1_len))
            - 1: second output tensor (2D (output2_num, output2_len))

    Attributes:
        required attrs (Dict[str, Any]): Expected keys
            - 'output1_num' (int): number of rows for output 1.
            - 'output1_len' (int): vector length of output 1.
            - 'output2_num' (int): number of rows for output 2.
            - 'output2_len' (int): vector length of output 2.

    Note:
        Cell contents remain intact during split (e.g. (2, 24) -> (2, 16) + (2, 8)), so
        distributions such as (2, 12) + (2, 12) that would reorder within a cell are invalid.

    Methods:
        - infer: Validate single input and return two output signatures.
        - para_node / para_connection: Split has no parameter node.
        - gen_prim: Construct PrimSplit configuration for hardware execution.
    """

    def __init__(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, attrs)
        required_attrs = ['output1_num', 'output1_len', 'output2_num', 'output2_len']
        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for Split operation.")
        self.primitive = True  # Marking as a non-primitive operation
    
    
    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        if len(inputs) != 1:
            raise ValueError("Split operation requires exactly 1 input.")

        input_data = inputs[0]

        output1_data_dtype = input_data.dtype
        output2_data_dtype = input_data.dtype
        output1_data_num = self.attrs['output1_num']
        output1_data_len = self.attrs['output1_len']
        output2_data_num = self.attrs['output2_num']
        output2_data_len = self.attrs['output2_len']
        output1_data_shape = (output1_data_num, output1_data_len)
        output2_data_shape = (output2_data_num, output2_data_len)
        return [(output1_data_shape, output1_data_dtype), (output2_data_shape, output2_data_dtype)]
    
    
    def para_node(self, inputs: Dict[int, Data], outputs: Dict[int, Data]) -> Optional[Data]:
        """Create a parameter node for Split."""
        return None
    
    def para_connection(self) -> bool:
        '''
        True: double connection (input and output)
        False: single connection (only input)
        '''
        return False
    
    def gen_prim(self, inputs: Dict[int, Data], outputs: Dict[int, Data], deps=0b00000000) -> intbv:
        input_data = inputs[0]
        output1_data = outputs[0]
        output2_data = outputs[1]
        x_addr = input_data.inferred_memref.addr if isinstance(input_data, ViewData) else input_data.memref.addr
        
        primSplit = PrimSplit(
            deps=deps, # Placeholder for dependencies
            # input_addr=input_data.memref.addr,
            input_addr=x_addr,
            output1_addr=output1_data.memref.addr,
            output2_addr=output2_data.memref.addr,
            input_num=input_data.shape[0],
            output1_num=self.attrs['output1_num'],
            output2_num=self.attrs['output2_num'],
            # 只考虑BF16和INT8吗？
            input_len_cell=(math.ceil(input_data.shape[1]/16) if input_data.dtype == DataType.BF16 else
                            (math.ceil(input_data.shape[1]/32) if input_data.dtype == DataType.INT8 else
                            math.ceil(input_data.shape[1]/256))),
            output1_len_cell=(math.ceil(output1_data.shape[1]/16) if output1_data.dtype == DataType.BF16 else
                            (math.ceil(output1_data.shape[1]/32) if output1_data.dtype == DataType.INT8 else
                            math.ceil(output1_data.shape[1]/256))),
            output2_len_cell=(math.ceil(output2_data.shape[1]/16) if output2_data.dtype == DataType.BF16 else
                            (math.ceil(output2_data.shape[1]/32) if output2_data.dtype == DataType.INT8 else
                            math.ceil(output2_data.shape[1]/256)))
        )
        
        return primSplit.PIC
        