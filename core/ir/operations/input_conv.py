from myhdl import bin, intbv
from typing import Any, Dict, List, Optional, Tuple
from core.ir.operation import Operation
from core.ir.data import Data, DataType, MemBlock
from core.ir.prims.input_convolution import PrimInputConvolution

class InputConv(Operation):
    """Input Convolution operation.

    Attributes:
        name (str): Human-readable identifier for the operation.
        attrs (Dict[str, Any]): Operation parameters dictionary.
            Expected keys in attrs:
                - 'kernel_size' (Tuple[int, int]): Size of the ORIGINAL convolution kernel.
                - 'stride' (Tuple[int, int]): Stride of the ORIGINAL convolution.
                - 'padding' (Tuple[int, int, int, int]): Padding added to both sides of the input.
                - 'dilation' (Tuple[int, int]): Spacing between kernel elements.
                - 'in_channels' (int): Number of input channels.
                - 'out_channels' (int): Number of output channels.

    Methods:
        
    """

    def __init__(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, attrs)
        required_attrs = ['kernel_size', 'stride', 'padding', 'dilation', 'in_channels', 'out_channels', 'padding_value']
        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for InputConvolution operation.")
        self.primitive = True  # Marking as a non-primitive operation
    
    
    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        raise NotImplementedError("Inference not implemented for DeepConv operation.")
    
    
    def para_node(self, inputs: Dict[int, Data], outputs: Dict[int, Data]) -> Optional[Data]:
        """Create a parameter node for deep convolution."""
        input_data = inputs[0]
        output_data = outputs[0]
        weight_data = inputs[1]
        bias_data = inputs[2] if 2 in inputs else None
        
        primConv = PrimInputConvolution(
            conv_type='INT8' if weight_data.dtype == DataType.INT8 else 'BF16',
            isSNN=(input_data.dtype == DataType.SPIKE),
            deps=0b00000000, # Placeholder for dependencies
            x_in_h=input_data.shape[1],
            x_in_w=input_data.shape[2],
            c_in=input_data.shape[3],
            c_out=output_data.shape[3],
            k_h=self.attrs['kernel_size'][0],
            k_w=self.attrs['kernel_size'][1],
            x_in_addr=input_data.memref.addr,
            w_in_addr=weight_data.memref.addr,
            b_in_addr=bias_data.memref.addr if bias_data else 0,
            y_out_addr=output_data.memref.addr,
            bs=input_data.shape[0],
            dilation_h=self.attrs['dilation'][0],
            dilation_w=self.attrs['dilation'][1],
            stride_h=self.attrs['stride'][0],
            stride_w=self.attrs['stride'][1],
            padding_top=self.attrs['padding'][0],
            padding_bottom=self.attrs['padding'][1],
            padding_left=self.attrs['padding'][2],
            padding_right=self.attrs['padding'][3],
            padding_value=self.attrs['padding_value'],
            param_addr_1=0,  # Placeholder
            param_addr_2=0,  # Placeholder
            b_in_data=bias_data.payload if bias_data else None,
            x_in_data=input_data.payload,
            w_in_data=weight_data.payload
        )
        
        # TODO: Implement parameter node creation logic
        para_code = MemBlock(length=2, payload=[primConv.param1, primConv.param2])
        para_data = Data(name=f"{self.name}.params", memref=para_code)
        return para_data
    
    def para_connection(self) -> bool:
        '''
        True: double connection (input and output)
        False: single connection (only input)
        '''
        return False
    
    def gen_prim(self, inputs: Dict[int, Data], outputs: Dict[int, Data], deps=0b00000000) -> intbv:
        input_data = inputs[0]
        output_data = outputs[0]
        weight_data = inputs[1]
        bias_data = inputs[2] if 2 in inputs else None
        para_data = inputs[3]
        
        primConv = PrimInputConvolution(
            conv_type='INT8' if weight_data.dtype == DataType.INT8 else 'BF16',
            isSNN=(input_data.dtype == DataType.SPIKE),
            deps=deps,
            x_in_h=input_data.shape[1],
            x_in_w=input_data.shape[2],
            c_in=input_data.shape[3],
            c_out=output_data.shape[3],
            k_h=self.attrs['kernel_size'][0],
            k_w=self.attrs['kernel_size'][1],
            x_in_addr=input_data.memref.addr,
            w_in_addr=weight_data.memref.addr,
            b_in_addr=bias_data.memref.addr if bias_data else 0,
            y_out_addr=output_data.memref.addr,
            bs=input_data.shape[0],
            dilation_h=self.attrs['dilation'][0],
            dilation_w=self.attrs['dilation'][1],
            stride_h=self.attrs['stride'][0],
            stride_w=self.attrs['stride'][1],
            padding_top=self.attrs['padding'][0],
            padding_bottom=self.attrs['padding'][1],
            padding_left=self.attrs['padding'][2],
            padding_right=self.attrs['padding'][3],
            padding_value=self.attrs['padding_value'],
            param_addr_1=para_data.memref.addr,
            param_addr_2=para_data.memref.addr + 1,  # Placeholder
            b_in_data=bias_data.payload if bias_data else None,
            x_in_data=input_data.payload,
            w_in_data=weight_data.payload
        )
        
        return primConv.PIC
        