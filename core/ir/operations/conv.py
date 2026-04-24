from myhdl import bin, intbv
from typing import Any, Dict, List, Optional, Tuple, Union
from core.ir.operation import Operation
from core.ir.data import Data, DataType, ViewData
from core.ir.graph import Graph
from core.ir.operations.deep_conv import DeepConv

class Conv(Operation):
    """Convolution operation.

    Attributes:
        name (str): Human-readable identifier for the operation.
        attrs (Dict[str, Any]): Operation parameters dictionary.
            Expected keys in attrs:
                - 'kernel_size' (Tuple[int, int]): Size of the convolution kernel.
                - 'stride' (Tuple[int, int]): Stride of the convolution.
                - 'padding' (Tuple[int, int, int, int]): Padding added to 4 sides of the input.
                - 'dilation' (Tuple[int, int]): Spacing between kernel elements.
                - 'in_channels' (int): Number of input channels.
                - 'out_channels' (int): Number of output channels.

    Methods:
        infer(inputs: List[Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[str]]]:
            Infers the output shape and dtype based on input metadata and operation attributes.
        codegen(inputs: List[Data]) -> str:
            Generates a textual representation of the operation for code generation purposes.
    """

    def __init__(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, attrs)
        required_attrs = ['kernel_size', 'stride', 'padding', 'dilation', 'in_channels', 'out_channels', 'input_dtype', 'weight_dtype']
        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for DeepConvolution operation.")
        
        if "padding_value" not in self.attrs:
            self.attrs["padding_value"] = 0  # Default padding value
        if len(self.attrs["padding"]) != 4:
            if len(self.attrs["padding"]) == 2:
                self.attrs["padding"] = (self.attrs["padding"][0], self.attrs["padding"][0], self.attrs["padding"][1], self.attrs["padding"][1])
            elif len(self.attrs["padding"]) == 1:
                self.attrs["padding"] = (self.attrs["padding"][0], self.attrs["padding"][0], self.attrs["padding"][0], self.attrs["padding"][0])
            else:
                raise ValueError("Padding must be an int or a tuple of 2 or 4 ints.")
        
        possible_dtype_pairs = [
            (DataType.BF16, DataType.BF16),
            (DataType.INT8, DataType.INT8),
            (DataType.SPIKE, DataType.BF16),
            (DataType.SPIKE, DataType.INT8),
        ]
        
        if (self.attrs['input_dtype'], self.attrs['weight_dtype']) not in possible_dtype_pairs:
            raise ValueError(f"Invalid dtype combination: input_dtype={self.attrs['input_dtype']}, weight_dtype={self.attrs['weight_dtype']}. Allowed combinations are {possible_dtype_pairs}.")
        
        self.primitive = False  # Marking as a non-primitive operation
        
    
    def infer(self, inputs: Dict[int, Union[Data, ViewData]]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        if len(inputs) != 1:
            raise ValueError("DeepConvolution operation requires exactly 1 input.")
        assert list(inputs.keys()) == [0], "DeepConvolution operation only supports input on port 0."

        input_data = inputs[0]
        if input_data.shape is None or len(input_data.shape) != 4:
            raise ValueError("Input data must have shape (N, H_in, W_in, C_in).")

        N, H_in, W_in, C_in = input_data.shape
        assert C_in == self.attrs['in_channels'], "Input channels do not match 'in_channels' attribute."
        kernel_h, kernel_w = self.attrs['kernel_size']
        stride_h, stride_w = self.attrs['stride']
        pad_t, pad_b, pad_l, pad_r = self.attrs['padding']
        dil_h, dil_w = self.attrs['dilation']
        C_out = self.attrs['out_channels']
        
        assert input_data.dtype == self.attrs['input_dtype'], "Input dtype does not match 'input_dtype' attribute."

        H_out = ((H_in + pad_t + pad_b - dil_h * (kernel_h - 1) - 1) // stride_h) + 1
        W_out = ((W_in + pad_l + pad_r - dil_w * (kernel_w - 1) - 1) // stride_w) + 1
        output_shape = (N, H_out, W_out, C_out)
        return [(output_shape, DataType.BF16)]
    
    
    def to_prim(self):
        subgraph = Graph()
        
        subgraph.add_node(DeepConv(self.name, self.attrs))
        weight_name = f"{self.name}.weights"
        bias_name = f"{self.name}.bias"
        subgraph.add_node(Data(weight_name, shape=(self.attrs['in_channels'], *self.attrs['kernel_size'], self.attrs['out_channels']), dtype=self.attrs['weight_dtype']))
        
        bias_type = DataType.BF16 if self.attrs['weight_dtype'] == DataType.BF16 else DataType.INT32
        subgraph.add_node(Data(bias_name, shape=(self.attrs['out_channels'],), dtype=bias_type))
        subgraph.connect(weight_name, self.name, 0, 1)
        subgraph.connect(bias_name, self.name, 0, 2)
        return {
            "subgraph": subgraph, 
            "input_mapping": {0: (self.name, 0)}, 
            "output_mapping": {0: (self.name, 0)}
        }
        