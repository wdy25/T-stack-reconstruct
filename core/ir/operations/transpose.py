from myhdl import bin, intbv
from typing import Any, Dict, List, Optional, Tuple
from core.ir.operation import Operation
from core.ir.data import Data, DataType, MemBlock, ViewData
from core.ir.prims.transposition import PrimTransposition

class Transpose(Operation):
    """Transposition operation.

    Ports:
        inputs:
            - 0: Input tensor to be transposed.(4D (dim_A, dim_B, dim_C, dim_D))
        outputs:
            - 0: Transposed output tensor.(the same rank as inputs[0])

    Attributes:
        name (str): Human-readable identifier for the operation.
        attrs (Dict[str, Any]): Operation parameters dictionary.
            Expected keys in attrs:
                - 'dim_A' (int): size of dimension A.
                - 'dim_B' (int): size of dimension B.
                - 'dim_C' (int): size of dimension C.
                - 'dim_D' (int): size of dimension D.
                - 'transpose_order' (str): "AB" or "AC" or "AD" or "BC" or "BD" or "CD".

    Methods:
        
    """

    def __init__(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, attrs)
        required_attrs = ['dim_A', 'dim_B', 'dim_C', 'dim_D', 'transpose_order']
        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for Transpose operation.")
        self.primitive = True  # Marking as a non-primitive operation
    
    
    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        input_data = inputs[0]
        dima, dimb, dimc, dimd = input_data.shape
        assert dima == self.attrs['dim_A'], f"Input shape mismatch for Transpose operation. Expected dimA: {self.attrs['dim_A']}, got {dima}."
        assert dimb == self.attrs['dim_B'], f"Input shape mismatch for Transpose operation. Expected dimB: {self.attrs['dim_B']}, got {dimb}."
        assert dimc == self.attrs['dim_C'], f"Input shape mismatch for Transpose operation. Expected dimC: {self.attrs['dim_C']}, got {dimc}."
        assert dimd == self.attrs['dim_D'], f"Input shape mismatch for Transpose operation. Expected dimD: {self.attrs['dim_D']}, got {dimd}."
        permute_order = self.attrs['transpose_order'] # e.g. (0,1,3,2)
        # permute dimensions
        if permute_order == "CD":
            new_shape = (dima, dimb, dimd, dimc)
        elif permute_order == "BD":
            new_shape = (dima, dimd, dimc, dimb)
        elif permute_order == "AD":
            new_shape = (dimd, dimb, dimc, dima)
        elif permute_order == "BC":
            new_shape = (dima, dimc, dimb, dimd)
        elif permute_order == "AC":
            new_shape = (dimc, dimb, dima, dimd)
        elif permute_order == "AB":
            new_shape = (dimb, dima, dimc, dimd)
        else:
            raise ValueError(f"Invalid transpose order: {permute_order}")
        
        return [(new_shape, input_data.dtype)]

    
    def para_node(self, inputs: Dict[int, Data], outputs: Dict[int, Data]) -> Optional[Data]:
        """Create a parameter node for deep convolution."""
        input_data = inputs[0]
        output_data = outputs[0]
        # address isn't needed when creating parameter node, so just set 0.
        x_addr = 0
        y_out_addr = 0
        
        
        primConv = PrimTransposition(
            transpose_type='INT8' if input_data.dtype == DataType.INT8 else 'BF16',
            transpose_order=self.attrs['transpose_order'],
            deps=0b00000000, # Placeholder for dependencies
            dim_A=input_data.shape[0],
            dim_B=input_data.shape[1],
            dim_C=input_data.shape[2],
            dim_D=input_data.shape[3],
            x_in_addr=0,
            y_out_addr=0,
            param_addr_1=0,  # Placeholder
            # x_in_data=input_data.payload,
        )
        
        # TODO: Implement parameter node creation logic
        para_code = MemBlock(length=1, payload=[primConv.param1])
        para_data = Data(name=f"{self.name}.params", memref=para_code)
        return para_data
    
    def para_connection(self) -> bool:
        '''
        True: double connection (input and output)
        False: single connection (only input)
        '''
        return False
    
    def gen_prim(self, inputs: Dict[int, Data], outputs: Dict[int, Data], deps=0b00000000) -> intbv:
        primitive = self.build_prim(inputs, outputs, deps)
        return primitive.PIC

    def build_prim(self, inputs: Dict[int, Data], outputs: Dict[int, Data], deps=0b00000000) -> PrimTransposition:
        input_data = inputs[0]
        output_data = outputs[0]
        para_data = inputs[1]
        x_addr = input_data.inferred_memref.addr if isinstance(input_data, ViewData) else input_data.memref.addr
        
        primitive = PrimTransposition(
            transpose_type='INT8' if input_data.dtype == DataType.INT8 else 'BF16',
            transpose_order=self.attrs['transpose_order'],
            deps=deps,
            dim_A=input_data.shape[0],
            dim_B=input_data.shape[1],
            dim_C=input_data.shape[2],
            dim_D=input_data.shape[3],
            x_in_addr=x_addr,
            y_out_addr=output_data.memref.addr,
            param_addr_1=para_data.memref.addr,
            # x_in_data=input_data.payload,
        )
        
        return primitive
        
