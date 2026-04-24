from myhdl import bin, intbv
from typing import Any, Dict, List, Optional, Tuple
from core.ir.operation import Operation
from core.ir.data import Data, DataType, MemBlock, ViewData, ConcatData
from core.ir.prims.matrix_multiplication import PrimMatrixMultiplication

class MatMul(Operation):
    """Matrix Multiplication operation.

    Ports:
        inputs:
            - 0: input tensor 3D (Dim_A, Batch_size, C_in) or 2D (Batch_size, C_in)
            - 1: weight tensor 3D (Dim_A, C_in, C_out) or 2D (C_in, C_out).
            - 2: bias tensor 2D (1, C_out) or 1D (C_out,)
                - input and weight tensor can be either both 3D or both 2D, or input 3D and weight 2D.
                    - When input is 3D and weight is 2D, treat the input as (Dim_A * Batch_size, C_in) for matrix multiplication. like: input: (Dim_A, Batch_size, C_in) -> reshape: (Dim_A * Batch_size, C_in) -> output: (Dim_A * Batch_size, C_out) -> reshape back to (Dim_A, Batch_size, C_out)
        outputs:
            - 0: output tensor 3D (Dim_A, Batch_size, C_out) or 2D (Batch_size, C_out). the rank is same as input tensor.

    Attributes:
        name (str): Human-readable identifier for the operation.
        attrs (Dict[str, Any]): Operation parameters dictionary.
            Expected keys in attrs:
                - 'dim_A' (int): Number of slices in the first dimension. only required when input and weight are both 3D.
                - 'in_channels' (int): Number of input channels.
                - 'out_channels' (int): Number of output channels.
                - 'batch_size' (int): Batch size.

    Supported Data Type Combinations:
        The operation supports the following combinations of input, weight, and bias data types:
            - input: INT8, weight: INT8, bias: INT32
            - input: SPIKE, weight: INT8, bias: INT32
            - input: BF16, weight: BF16, bias: BF16
            - input: SPIKE, weight: BF16, bias: BF16
            - input: SPIKE, weight: SPIKE, bias: INT32
        
    """

    def __init__(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, attrs)
        # required_attrs = ['dim_A', 'in_channels', 'out_channels', 'batch_size']
        required_attrs = []
        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for MatrixMultiplication operation.")
        self.primitive = True  # Marking as a non-primitive operation
    
    
    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        # raise NotImplementedError("Inference not implemented for MatMul operation.")
        input_data = inputs[0]
        if input_data.shape is None or not(len(input_data.shape) in (2,3)):
            raise ValueError("Input data must have shape (Dim_A, Batch_size, C_in) or (Batch_size, C_in).")
        weight_data = inputs[1]
        if weight_data.shape is None or not(len(weight_data.shape) in (2,3)):
            raise ValueError("Weight data must have shape (Dim_A, C_in, C_out) or (C_in, C_out).")
        bias_data = inputs[2]
        if bias_data.shape is None or not(len(bias_data.shape) in (1,2)):
            raise ValueError("Bias data must have shape (1, C_out) or (C_out,).")
        if len(bias_data.shape) == 2 and bias_data.shape[0] != 1:
            raise ValueError("Bias data must have shape (1, C_out) when len(shape) is 2.")

        input_dims = len(input_data.shape)
        weight_dims = len(weight_data.shape)
        if not ((input_dims == 3 and weight_dims in (2, 3)) or (input_dims == 2 and weight_dims == 2)):
            raise ValueError("Unsupported combination of input and weight tensor ranks for MatMul.")

        assert input_data.shape[-1] == weight_data.shape[-2], "Input C_in does not match weight C_in."
        if input_dims == 3 and weight_dims == 3:
            assert input_data.shape[0] == weight_data.shape[0], "Input dim_A does not match weight dim_A."

        dim_A = self.attrs.get('dim_A', None)
        if dim_A is not None:
            assert dim_A == input_data.shape[0], "Input dim_A does not match 'dim_A' attribute."
            assert dim_A == weight_data.shape[0], "Weight dim_A does not match 'dim_A' attribute."
        c_in = self.attrs.get('in_channels', None)
        if c_in is not None:
            assert c_in == input_data.shape[-1], "Input C_in does not match 'in_channels' attribute."
            assert c_in == weight_data.shape[-2], "Weight C_in does not match 'in_channels' attribute."
        c_out = self.attrs.get('out_channels', None)
        if c_out is not None:
            assert c_out == weight_data.shape[-1], "Weight C_out does not match 'out_channels' attribute."
            assert c_out == bias_data.shape[-1], "Bias C_out does not match 'out_channels' attribute."
        batch_size = self.attrs.get('batch_size', None)
        if batch_size is not None:
            assert batch_size == input_data.shape[-2], "Input batch size does not match 'batch_size' attribute."

        if input_dims == 3 and weight_dims == 2:
            # mark that input is reshaped internally
            self.input_reshaped = True
        else:
            self.input_reshaped = False

        if input_dims == 3:
            output_shape = (input_data.shape[0], input_data.shape[1], weight_data.shape[-1])
        elif input_dims == 2:
            output_shape = (input_data.shape[0], weight_data.shape[-1])
        return [(output_shape, DataType.BF16)]
    
    
    def para_node(self, inputs: Dict[int, Data], outputs: Dict[int, Data]) -> Optional[Data]:
        """Create a parameter node for deep convolution."""
        input_data = inputs[0]
        output_data = outputs[0]
        weight_data = inputs[1]
        bias_data = inputs[2]
        # address isn't needed when creating parameter node, so just set 0.
        w_addr = 0
        x_addr = 0
        b_addr = 0
        y_out_addr = 0

        if self.input_reshaped:
            # when input is reshaped internally, we need to adjust dim_A and batch_size
            dim_A = 1
            batch_size = input_data.shape[0] * input_data.shape[1]
        elif len(input_data.shape) == 3:
            dim_A = input_data.shape[0]
            batch_size = input_data.shape[1]
        else:
            dim_A = 1
            batch_size = input_data.shape[0]
        
        if weight_data.dtype == DataType.INT8:
            matmul_type = 'INT8'
        elif weight_data.dtype == DataType.BF16:
            matmul_type = 'BF16'
        elif weight_data.dtype == DataType.SPIKE:
            matmul_type = 'SPIKE'
        else:
            raise ValueError("Unsupported weight data type for MatMul.")

        primConv = PrimMatrixMultiplication(
            matmul_type=matmul_type,
            isSNN=(input_data.dtype == DataType.SPIKE),
            deps=0b00000000, # Placeholder for dependencies
            c_in=input_data.shape[-1],
            dim_A=dim_A,
            c_out=output_data.shape[-1],
            # x_in_addr=input_data.memref.addr,
            x_in_addr=0,
            w_in_addr=0,
            # w_in_addr=weight_data.memref.addr,
            b_in_addr=0,
            y_out_addr=0,
            batch_size=batch_size,
            param_addr_1=0,  # Placeholder
            # b_in_data=bias_data.payload if bias_data else None,
            # x_in_data=input_data.payload,
            # w_in_data=weight_data.payload
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
        primitive = self.build_primitive(inputs, outputs, deps)
        return primitive.PIC

    def build_prim(
        self,
        inputs: Dict[int, Data],
        outputs: Dict[int, Data],
        deps: intbv = intbv(0)[8:],
    ) -> PrimMatrixMultiplication:
        """Build the primitive operation object."""
        input_data = inputs[0]
        output_data = outputs[0]
        weight_data = inputs[1]
        bias_data = inputs[2] if 2 in inputs else None
        para_data = inputs[3]
        w_addr = weight_data.inferred_memref.addr if isinstance(weight_data, (ViewData, ConcatData)) else weight_data.memref.addr
        x_addr = input_data.inferred_memref.addr if isinstance(input_data, (ViewData, ConcatData)) else input_data.memref.addr
        b_addr = bias_data.inferred_memref.addr if isinstance(bias_data, (ViewData, ConcatData)) else bias_data.memref.addr

        if self.input_reshaped:
            # when input is reshaped internally, we need to adjust dim_A and batch_size
            dim_A = 1
            batch_size = input_data.shape[0] * input_data.shape[1]
        elif len(input_data.shape) == 3:
            dim_A = input_data.shape[0]
            batch_size = input_data.shape[1]
        else:
            dim_A = 1
            batch_size = input_data.shape[0]
        
        if weight_data.dtype == DataType.INT8:
            matmul_type = 'INT8'
        elif weight_data.dtype == DataType.BF16:
            matmul_type = 'BF16'
        elif weight_data.dtype == DataType.SPIKE:
            matmul_type = 'SPIKE'
        else:
            raise ValueError("Unsupported weight data type for MatMul.")
        
        return PrimMatrixMultiplication(
            matmul_type=matmul_type,
            isSNN=(input_data.dtype == DataType.SPIKE),
            deps=deps,
            c_in=input_data.shape[-1],
            dim_A=dim_A,
            c_out=output_data.shape[-1],
            x_in_addr=x_addr,
            w_in_addr=w_addr,
            b_in_addr=b_addr,
            y_out_addr=output_data.memref.addr,
            batch_size=batch_size,
            param_addr_1=para_data.memref.addr,
        )
        