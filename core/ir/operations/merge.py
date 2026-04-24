from myhdl import intbv
from typing import Any, Dict, List, Optional, Tuple, Union
from core.ir.operation import Operation
from core.ir.data import Data, DataType, ViewData
from core.ir.prims.merge import PrimMerge
import math

TensorLike = Union[Data, ViewData]
_SUPPORTED_DTYPES = (DataType.BF16, DataType.INT8)
_CELL_WIDTH = {
    DataType.BF16: 16,
    DataType.INT8: 32,
}

class Merge(Operation):
    """Merge two tensors along the row dimension respecting cell boundaries.

    Ports:
        inputs:
            - 0: first input tensor (2D (input1_num, input1_len), BF16/INT8 supported)
            - 1: second input tensor (2D (input2_num, input2_len), BF16/INT8 supported)
        outputs:
            - 0: merged output tensor (2D (output_num, output_len))

    Attributes:
        required attrs (Dict[str, Any]): Expected keys
            - 'output_num' (int): total rows after merge.
            - 'output_len' (int): vector length of the merged tensor.

    Note:
        1. Cell contents remain intact during merge (e.g. (2, 16)+(2, 8) -> (2, 24)), so
        combinations such as (2, 12)+(2, 12) that would reorder within a cell are invalid.
        2. input1_num can be different from input2_num. for example, (2,32)+(4,16) -> pad (2,32) with 0 to (4,32)
        -> (4,32)+(4,16) -> (4,48)

    Methods:
        - infer: Ensure two inputs and return a single output signature.
        - para_node / para_connection: Merge has no parameter node.
        - gen_prim: Construct PrimMerge configuration for hardware execution.
    """

    def __init__(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, attrs)
        required_attrs = ['output_num', 'output_len']
        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for Merge operation.")
        self.primitive = True  # Marking as a primitive operation
    
    
    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        if len(inputs) != 2:
            raise ValueError("Merge operation requires exactly 2 input.")

        input1_data = inputs[0]
        input2_data = inputs[1]

        assert input1_data.dtype == input2_data.dtype, "input data's type do not match."
        output_data_dtype = input1_data.dtype
        output_data_num = self.attrs['output_num']
        output_data_len = self.attrs['output_len']
        output_data_shape = (output_data_num, output_data_len)
        return [(output_data_shape, output_data_dtype)]
    
    
    def para_node(self, inputs: Dict[int, Data], outputs: Dict[int, Data]) -> Optional[Data]:
        """Create a parameter node for Merge."""
        return None
    
    def para_connection(self) -> bool:
        '''
        True: double connection (input and output)
        False: single connection (only input)
        '''
        return False
    
    def gen_prim(self, inputs: Dict[int, Data], outputs: Dict[int, Data], deps=0b00000000) -> intbv:
        input1_data = inputs[0]
        input2_data = inputs[1]
        output_data = outputs[0]

        input1_num = 1
        for i in range(len(input1_data.shape)-1):
            input1_num = input1_num * input1_data.shape[i]
        input2_num = 1
        for i in range(len(input2_data.shape)-1):
            input2_num = input2_num * input2_data.shape[i]
        
        primMerge = PrimMerge(
            deps=deps, # Placeholder for dependencies
            input1_addr=input1_data.memref.addr,
            input2_addr=input2_data.memref.addr,
            output_addr=output_data.memref.addr,
            input1_num=input1_num,
            input2_num=input2_num,
            output_num=self.attrs['output_num'],
            # 只考虑BF16和INT8吗？
            input1_len_cell=(math.ceil(input1_data.shape[-1]/16) if input1_data.dtype == DataType.BF16 else
                            (math.ceil(input1_data.shape[-1]/32) if input1_data.dtype == DataType.INT8 else
                            math.ceil(input1_data.shape[-1]/256))),
            input2_len_cell=(math.ceil(input2_data.shape[-1]/16) if input2_data.dtype == DataType.BF16 else
                            (math.ceil(input2_data.shape[-1]/32) if input2_data.dtype == DataType.INT8 else
                            math.ceil(input2_data.shape[-1]/256))),
            output_len_cell=(math.ceil(output_data.shape[-1]/16) if output_data.dtype == DataType.BF16 else
                            (math.ceil(output_data.shape[-1]/32) if output_data.dtype == DataType.INT8 else
                            math.ceil(output_data.shape[-1]/256)))
        )
        
        return primMerge.PIC
        