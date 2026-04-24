import math
from typing import Any, Dict, List, Optional, Tuple, Union

from myhdl import intbv

from core.ir.operation import Operation
from core.ir.data import Data, DataType, MemBlock, elements_to_32b_cell
from core.ir.prims.maxpooling import PrimPooling


class Clamp(Operation):
    """Clamp operation to limit the values of a tensor (It's created by 1x1 MaxPooling).

    Ports:
        inputs:
            - 0: input tensor (>=1D, BF16 only)
            - 1: para
        outputs:
            - 0: output tensor (the same shape as inputs[0], BF16 or INT8)

    Attributes:
        name (str): Human-readable identifier for the operation.
        required attrs (Dict[str, Any]): Operation parameters dictionary. Expected keys:
            - 'clamp_type' (str): 'max' for max clamping, 'min' for min clamping.
            - 'threshold' (int, float): 
                - when clamp_type is 'max', values above this threshold will be clamped. It is equivalent to ``min(x, threshold)``.
                - when clamp_type is 'min', values below this threshold will be clamped. It is equivalent to ``max(x, threshold)``.
            - 'output_dtype' (DataType): Desired output tensor type.
    """

    def __init__(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, attrs)

        required_attrs = ["clamp_type", "threshold"]
        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for Clamp operation.")

        clamp_type = str(self.attrs["clamp_type"]).lower()
        if clamp_type not in ("max", "min"):
            raise ValueError("'clamp_type' must be 'max' or 'min'.")
        self.attrs["clamp_type"] = clamp_type

        threshold = self.attrs["threshold"]
        if not isinstance(threshold, (int, float)):
            raise ValueError("'threshold' must be a numeric type.")
        threshold_value = float(threshold)
        self.attrs["threshold"] = threshold_value

        output_dtype = self.attrs["output_dtype"]
        if output_dtype not in (DataType.BF16, DataType.INT8):
            raise ValueError("'output_dtype' must be DataType.BF16 or DataType.INT8.")

        if clamp_type == "max":
            self.attrs["pool_type"] = "min"
        elif clamp_type == "min":
            self.attrs["pool_type"] = "max"
        self.pool_type_number = 0 if self.attrs["pool_type"] == "max" else 1

        # Pre-configured internal attributes compatible with MaxPooling primitive
        self.attrs["kernel_size"] = (1, 1)
        self.attrs["stride"] = (1, 1)
        self.attrs["scaler"] = 0
        self.attrs["scaler_mode"] = 0
        self.attrs["bias_mode"] = 1
        self.attrs["bias_scalar"] = threshold_value

        self.primitive = True

    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        # requires input on port 0
        input_data = inputs[0]

        if input_data.shape is None or len(input_data.shape) < 1:
            raise ValueError("Clamp input tensor rank must be at least 1.")

        if input_data.dtype != DataType.BF16:
            raise ValueError("Clamp currently only supports BF16 input tensors.")

        output_dtype = self.attrs["output_dtype"]

        return [(input_data.shape, output_dtype)]

    def para_node(self, inputs: Dict[int, Data], outputs: Dict[int, Data]) -> Optional[Data]:
        # para_node requires input and output on port 0
        input_data = inputs[0]

        threshold = self.attrs["threshold"]

        prim_clamp = self._build_primitive(
            input_data=input_data,
            x_in_addr=0,
            y_out_addr=0,
            para_addr=0,
            deps=0b00000000,
            threshold=threshold, # actually threshold don't be needed in getting para prim
        )

        para_code = MemBlock(length=1, payload=[prim_clamp.para])
        para_data = Data(name=f"{self.name}.params", memref=para_code)
        return para_data

    def para_connection(self) -> bool:
        '''
        True: double connection (input and output)
        False: single connection (only input)
        '''
        return False

    def gen_prim(
        self,
        inputs: Dict[int, Data],
        outputs: Dict[int, Data],
        deps: intbv = intbv(0)[8:],
    ) -> intbv:
        # gen_prim requires input and output on port 0
        input_data = inputs[0]
        output_data = outputs[0]
        
        # gen_prim requires para on last port(port 2 or port1 when no bias) 
        para_data = inputs[2] if self.attrs["bias_mode"] == 2 else inputs[1]

        x_in_addr = input_data.memref.addr
        y_out_addr = output_data.memref.addr
        para_addr = para_data.memref.addr

        threshold = self.attrs["threshold"]
        
        prim_clamp = self._build_primitive(
            input_data=input_data,
            x_in_addr=x_in_addr,
            y_out_addr=y_out_addr,
            para_addr=para_addr,
            deps=deps,
            threshold=threshold,
        )

        return prim_clamp.PIC

    def _build_primitive(
        self,
        input_data: Data,
        x_in_addr: int,
        y_out_addr: int,
        para_addr: int,
        deps: intbv,
        threshold: Union[int, float],
    ) -> PrimPooling:

        batch_size, x_in_h, x_in_w, channels = self._expand_to_pooling_shape(input_data.shape)
        c_in_32b = elements_to_32b_cell(channels, input_data.dtype)

        kernel_h, kernel_w = self.attrs["kernel_size"]
        stride_h, stride_w = self.attrs["stride"]

        output_dtype = self.attrs.get("output_dtype", input_data.dtype)
        y_type = 1 if output_dtype == DataType.BF16 else 0

        return PrimPooling(
            deps=deps,
            x_in_addr=x_in_addr,
            bias_value_or_addr=threshold,  # threshold is the bias value when bias_mode is 1 in MaxPooling
            out_addr=y_out_addr,
            para_addr=para_addr,
            batch_size=batch_size - 1,
            x_in_h=x_in_h - 1,
            x_in_w=x_in_w - 1,
            c_in_32B=c_in_32b - 1,
            kernel_h=kernel_h - 1,
            kernel_w=kernel_w - 1,
            scaler=0,
            scaler_mode=0,
            max_or_min=self.pool_type_number,
            y_type=y_type,
            bias_mode=1,
            stride_h=stride_h - 1,
            stride_w=stride_w - 1,
        )

    def _expand_to_pooling_shape(self, shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        if len(shape) == 1:
            return (1, 1, 1, shape[0])
        if len(shape) >= 2:
            batch = math.prod(shape[:-1])
            return (batch, 1, 1, shape[-1])