from myhdl import bin, intbv
from typing import Any, Dict, List, Optional, Tuple
from core.ir.operation import Operation
from core.ir.data import Data, DataType, MemBlock

class PoolingReLU(Operation):
    """Max Pooling with ReLU activation operation.

    Attributes:
        name (str): Human-readable identifier for the operation.
        attrs (Dict[str, Any]): Operation parameters dictionary.
            Expected keys in attrs:
                - 'pool_size' (Tuple[int, int]): Size of the max pooling kernel.
                - 'stride' (Tuple[int, int]): Stride of the pooling operation.
                - 'output_dtype' (DataType): Output data type.

    Methods:
        infer(inputs: List[Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
            Infers the output shape and dtype based on input metadata and operation attributes.
        para_node() -> MemBlock:
            Create a parameter node for max pooling relu operation.
    """

    def __init__(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, attrs)
        required_attrs = ['pool_size', 'stride', 'output_dtype']
        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for PoolingRelu operation.")
        
        # Validate data types
        supported_dtypes = [DataType.BF16, DataType.INT8]
        if self.attrs['output_dtype'] not in supported_dtypes:
            raise ValueError(f"Unsupported output_dtype: {self.attrs['output_dtype']}")
        
        self.primitive = True  # Marking as a primitive operation
    
    
    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        """Infer output shape and data type for PoolingRelu operation."""
        assert len(inputs) == 1, "PoolingRelu operation requires exactly 1 input."
        assert list(inputs.keys()) == [0], "PoolingRelu operation only supports input on port 0."

        input_data = inputs[0]
        if input_data.shape is None or len(input_data.shape) != 4:
            raise ValueError("Input data must have shape (N, H_in, W_in, C_in).")

        N, H_in, W_in, C_in = input_data.shape
        pool_h, pool_w = self.attrs['pool_size']
        stride_h, stride_w = self.attrs['stride']
        
        # Validate input dtype
        assert input_data.dtype == DataType.BF16, "Input dtype does not match 'input_dtype' attribute."

        # Calculate output dimensions for max pooling
        H_out = ((H_in + 2 * 0 - pool_h) // stride_h) + 1
        W_out = ((W_in + 2 * 0 - pool_w) // stride_w) + 1
        
        # Output shape maintains the same number of channels
        output_shape = (N, H_out, W_out, C_in)
        
        # Return output shape and dtype
        return [(output_shape, self.attrs['output_dtype'])]
    
    
    def para_node(self, inputs: Dict[int, Data], outputs: Dict[int, Data]) -> Optional[Data]:
        """Create a parameter node for max pooling relu operation.
        
        For max pooling with ReLU, parameters might include bias terms
        for ReLU activation or quantization parameters for different data types.
        """
        # TODO: Implement parameter node creation logic
        para_code = MemBlock(length=1, payload=[intbv(val=0, min=0, max=(1<<256))])  # Placeholder
        para_data = Data(name=f"{self.name}.params", memref=para_code)
        return para_data
    
    def para_connection(self) -> bool:
        '''
        True: double connection (input and output)
        False: single connection (only input)
        '''
        return False
    
    def to_prim(self):
        """Convert to a primitive operation representation if needed."""
        return None