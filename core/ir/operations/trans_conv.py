from myhdl import bin, intbv
from typing import Any, Dict, List, Optional, Tuple
from core.ir.operation import Operation
from core.ir.data import Data, DataType, MemBlock
from core.ir.prims.trans_convolution import PrimTransConvolution

class TransConv(Operation):
    """Transposed Convolution operation.

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
        # required_attrs = ['kernel_size', 'stride', 'padding', 'dilation', 'in_channels', 'out_channels', 'padding_value']
        required_attrs = []
        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for TransConvolution operation.")
        self.primitive = True  # Marking as a non-primitive operation
    
    
    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        input_data = inputs[0]
        weight_data = inputs[1]
        bias_data = inputs[2]

        bs, xh, xw, cin = input_data.shape
        kh, kw, _, cout = weight_data.shape
        pad_top, pad_bottom, pad_left, pad_right = self.attrs['padding']
        stride_h, stride_w = self.attrs['stride']
        dil_h, dil_w = self.attrs['dilation']
        input_dil_h, input_dil_w = self.attrs['input_dilation']
        xh = xh + (xh - 1) * (input_dil_h - 1)
        xw = xw + (xw - 1) * (input_dil_w - 1)
        yh = xh + pad_top + pad_bottom - kh + 1
        yw = xw + pad_left + pad_right - kw + 1
        output_data_type = DataType.BF16

        return [((bs,yh,yw,cout), output_data_type)]
    
    
    def para_node(self, inputs: Dict[int, Data], outputs: Dict[int, Data]) -> Optional[Data]:
        """Create a parameter node for deep convolution."""
        input_data = inputs[0]
        output_data = outputs[0]
        weight_data = inputs[1]
        bias_data = inputs[2] if 2 in inputs else None
        
        primConv = PrimTransConvolution(
            conv_type='BF16',
            isSNN=0,
            deps=0b00000000, # Placeholder for dependencies
            x_in_h=input_data.shape[1],
            x_in_w=input_data.shape[2],
            c_in=input_data.shape[3],
            c_out=output_data.shape[3],
            k_h=self.attrs['kernel_size'][0],
            k_w=self.attrs['kernel_size'][1],
            x_in_addr=0,
            w_in_addr=0,
            b_in_addr=0,
            y_out_addr=0,
            bs=input_data.shape[0],
            dilation_h=1,
            dilation_w=1,
            stride_h=1,
            stride_w=1,
            input_dilation_h=self.attrs['input_dilation'][0],
            input_dilation_w=self.attrs['input_dilation'][1],
            padding_top=self.attrs['padding'][0],
            padding_bottom=self.attrs['padding'][1],
            padding_left=self.attrs['padding'][2],
            padding_right=self.attrs['padding'][3],
            padding_value=0,
            param_addr_1=0,  # Placeholder
            param_addr_2=0,  # Placeholder
            param_addr_3=0,  # Placeholder
        )
        
        # TODO: Implement parameter node creation logic
        para_code = MemBlock(length=3, payload=[primConv.param1, primConv.param2, primConv.param3])
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
        
        primConv = PrimTransConvolution(
            conv_type='BF16',
            isSNN=0,
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
            dilation_h=1,
            dilation_w=1,
            stride_h=1,
            stride_w=1,
            input_dilation_h=self.attrs['input_dilation'][0],
            input_dilation_w=self.attrs['input_dilation'][1],
            padding_top=self.attrs['padding'][0],
            padding_bottom=self.attrs['padding'][1],
            padding_left=self.attrs['padding'][2],
            padding_right=self.attrs['padding'][3],
            padding_value=0,
            param_addr_1=para_data.memref.addr,
            param_addr_2=para_data.memref.addr + 1,  # Placeholder
            param_addr_3=para_data.memref.addr + 2,  # Placeholder
        )
        
        return primConv.PIC
    
    def build_prim(self,
        inputs: Dict[int, Data],
        outputs: Dict[int, Data],
        deps: intbv = intbv(0)[8:],
    ):
        input_data = inputs[0]
        output_data = outputs[0]
        weight_data = inputs[1]
        bias_data = inputs[2] if 2 in inputs else None
        para_data = inputs[3]

        primitive = self._build_primitive(input_data, weight_data, bias_data, output_data, para_data, deps)
        return primitive

    def _build_primitive(
        self,
        input_data: Data,
        weight_data: Data,
        bias_data: Optional[Data],
        output_data: Data,
        para_data: Data,
        deps: intbv,
    ) -> PrimTransConvolution:

        return PrimTransConvolution(
            conv_type='BF16',
            isSNN=0,
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
            dilation_h=1,
            dilation_w=1,
            stride_h=1,
            stride_w=1,
            input_dilation_h=self.attrs['input_dilation'][0],
            input_dilation_w=self.attrs['input_dilation'][1],
            padding_top=self.attrs['padding'][0],
            padding_bottom=self.attrs['padding'][1],
            padding_left=self.attrs['padding'][2],
            padding_right=self.attrs['padding'][3],
            padding_value=0,
            param_addr_1=para_data.memref.addr,
            param_addr_2=para_data.memref.addr + 1,  # Placeholder
            param_addr_3=para_data.memref.addr + 2,  # Placeholder
        )
        