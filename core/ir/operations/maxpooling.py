from math import ceil
from typing import Any, Dict, List, Optional, Tuple

from myhdl import intbv

from core.ir.operation import Operation
from core.ir.data import Data, DataType, MemBlock, elements_to_32b_cell
from core.ir.prims.maxpooling import PrimPooling


class MaxPooling(Operation):
    """Maxpooling operation backed by :class:`PrimPooling`.

    Ports: 
        inputs:
            - 0: input tensor (4D (N, H_in, W_in, C_in), BF16 only)
            - 1: bias tensor (exit only when bias_mode == 2)
                - when bias_mode == 0: no bias, there should be no bias tensor coonected to input[1]
                - when bias_mode == 1: scalar bias(int, float), there should be no bias tensor coonected to input[1]
                - when bias_mode == 2: vector bias, bias tensor shape should be 4D (1, 1, 1, C_in)
            - 2: para
                - when bias_mode == 2: para in inputs[2], else para in input[1]
        outputs:
            - 0: output tensor (4D (N, H_out, W_out, C_in), BF16 or INT8)

    Attributes:
        name (str): Human-readable identifier for the operation.
        required attrs (Dict[str, Any]): Operation parameters dictionary. Expected keys:
            - 'kernel_size' (Tuple[int, int]): Size of the pooling kernel.
            - 'stride' (Tuple[int, int]): Stride applied during pooling.
            - 'scaler' (int): Integer scaling factor applied when `scaler_mode` enables it.
            - 'scaler_mode' (int): Hardware scaling mode select bit (see table below).
            - 'output_dtype' (DataType): Desired output tensor type.
            - 'bias_mode' (int): Bias comparison mode selector (see table below).
        optional attrs (Dict[str, Any]): Optional operation parameters. Expected keys:
            - 'bias_scalar' (int, float): Scalar bias value only be required when bias_mode == 1.
            - 'pool_type' (str): 'max' for max pooling, 'min' for min pooling. default is 'max'.

    Attribute bit usage (PrimPooling PIC field reference):
        - scaler_mode: width=1, bits=[167]; 0 disables scaling, 1 applies the provided `scaler` value.
        - bias_mode: width=2, bits=[172:173]; 
            - 0 means no bias comparison, 
            - 1 means to compare against scalar bias "a", 
            - 2 means to compare against vector bias "A". ReLU configuration uses only 1 (scalar compare).

    Methods:
        infer: Derives output tensor shape and dtype from the provided inputs.
        para_node: Packages a parameter block as `Data` for scheduling.
        para_connection: States whether the parameter stage is double- or single-connected.
        gen_prim: Builds the runtime PrimPooling instruction with dependencies.
        _build_primitive: Populates a PrimPooling instance from tensors and attrs.
        _resolve_bias_value_or_node: Resolves the bias literal or memory-backed node.
    """

    def __init__(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, attrs)
        required_attrs = [
            "kernel_size",
            "stride",
            "scaler",
            "scaler_mode",
            "output_dtype",
            "bias_mode",
        ]
        optional_attrs = [
            "bias_scalar",  # required if bias_mode == 1
            "pool_type",    # optional, selects max (default) or min pooling
        ]
        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for MaxPooling operation.")

        if not isinstance(self.attrs["kernel_size"], tuple) or len(self.attrs["kernel_size"]) != 2:
            raise ValueError("'kernel_size' must be a tuple of two integers.")
        if not isinstance(self.attrs["stride"], tuple) or len(self.attrs["stride"]) != 2:
            raise ValueError("'stride' must be a tuple of two integers.")
        if not all(isinstance(x, int) for x in self.attrs["kernel_size"]):
            raise ValueError("'kernel_size' entries must be integers.")
        if not all(isinstance(x, int) for x in self.attrs["stride"]):
            raise ValueError("'stride' entries must be integers.")

        supported_output_dtype = (DataType.BF16, DataType.INT8)
        if self.attrs["output_dtype"] not in supported_output_dtype:
            raise ValueError(
                f"Unsupported output_dtype {self.attrs['output_dtype']}. Supported: {supported_output_dtype}."
            )

        if self.attrs["bias_mode"] == 0 and self.attrs["kernel_size"] == (1, 1):
            raise ValueError("'bias_mode' = 0 and 'kernel_size' = (1, 1) can't be set at the same time.")
        
        if self.attrs["bias_mode"] == 1 and "bias_scalar" not in self.attrs:
            raise ValueError("'bias_scalar' is required when bias_mode is 1.")

        if self.attrs["bias_mode"] not in (0, 1, 2):
            raise ValueError("'bias_mode' must be 0, 1, or 2.")
        if self.attrs["scaler_mode"] not in (0, 1):
            raise ValueError("'scaler_mode' must be 0 or 1.")

        pool_type = self.attrs.get("pool_type", "max")
        assert pool_type in ("max", "min"), "'pool_type' must be 'max' or 'min'."
        if pool_type == "max":
            self.pool_type_number = 0
        else:
            self.pool_type_number = 1
        

        self.primitive = True # Marking as a primitive operation

    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        # MaxPooling requires input on port 0
        input_data = inputs[0]

        if input_data.shape is None or len(input_data.shape) != 4:
            raise ValueError("Input data must have shape (N, H_in, W_in, C_in).")
        
        if input_data.dtype != DataType.BF16:
            raise ValueError("MaxPooling currently supports only BF16 input tensors.")

        kernel_h, kernel_w = self.attrs["kernel_size"]
        stride_h, stride_w = self.attrs["stride"]

        N, H_in, W_in, C_in = input_data.shape
        if H_in < kernel_h or W_in < kernel_w:
            raise ValueError("Pooling kernel larger than input spatial dimensions.")

        H_out = ((H_in - kernel_h) // stride_h) + 1
        W_out = ((W_in - kernel_w) // stride_w) + 1

        # if self.attrs["output_dtype"] == DataType.BF16:
        #     C_out = C_in
        # elif self.attrs["output_dtype"] == DataType.INT8:
        #     C_out = int(ceil(C_in / 2))

        output_shape = (N, H_out, W_out, C_in)
        return [(output_shape, self.attrs["output_dtype"])]

    def para_node(self, inputs: Dict[int, Data], outputs: Dict[int, Data]) -> Optional[Data]:
        # MaxPooling para_node requires input and output on port 0
        input_data = inputs[0]

        prim_maxpooling = self._build_primitive(
            input_data=input_data,
            x_in_addr=0,
            y_out_addr=0,
            para_addr=0,
            deps=0b00000000,
            bias_value_or_addr=0, # actually bias don't be needed in getting para prim
        )

        para_code = MemBlock(length=1, payload=[prim_maxpooling.para])
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
        # MaxPooling gen_prim requires input and output on port 0
        input_data = inputs[0]
        output_data = outputs[0]

        # MaxPooling gen_prim requires bias on port 1 if bias_mode == 2
        bias_value_or_node = self._resolve_bias_value_or_node(inputs)
        
        # MaxPooling gen_prim requires para on last port(port 2 or port1 when no bias) 
        para_data = inputs[2] if self.attrs["bias_mode"] == 2 else inputs[1]

        x_in_addr = input_data.memref.addr
        y_out_addr = output_data.memref.addr
        para_addr = para_data.memref.addr

        if self.attrs["bias_mode"] == 2:
            bias_value_or_addr = bias_value_or_node.memref.addr # address
        else:
            bias_value_or_addr = bias_value_or_node # bias value (0 or scalar)
        
        prim_maxpooling = self._build_primitive(
            input_data=input_data,
            x_in_addr=x_in_addr,
            y_out_addr=y_out_addr,
            para_addr=para_addr,
            deps=deps,
            bias_value_or_addr=bias_value_or_addr,
        )

        return prim_maxpooling.PIC

    def build_prim(
        self,
        inputs: Dict[int, Data],
        outputs: Dict[int, Data],
        deps: intbv = intbv(0)[8:],
    ) -> PrimPooling:
        input_data = inputs[0]
        output_data = outputs[0]

        bias_value_or_node = self._resolve_bias_value_or_node(inputs)
        para_data = inputs[2] if self.attrs["bias_mode"] == 2 else inputs[1]

        if self.attrs["bias_mode"] == 2:
            bias_value_or_addr = bias_value_or_node.memref.addr
        else:
            bias_value_or_addr = bias_value_or_node

        return self._build_primitive(
            input_data=input_data,
            x_in_addr=input_data.memref.addr,
            y_out_addr=output_data.memref.addr,
            para_addr=para_data.memref.addr,
            deps=deps,
            bias_value_or_addr=bias_value_or_addr,
        )

    def _build_primitive(
        self,
        input_data: Data,
        x_in_addr: int,
        y_out_addr: int,
        para_addr: int,
        deps: intbv,
        bias_value_or_addr: Optional[Data],
    ) -> PrimPooling:

        batch_size = input_data.shape[0]
        x_in_h = input_data.shape[1]
        x_in_w = input_data.shape[2]
        c_in_32b = elements_to_32b_cell(input_data.shape[3], input_data.dtype)

        kernel_h, kernel_w = self.attrs["kernel_size"]
        stride_h, stride_w = self.attrs["stride"]

        y_type = 1 if self.attrs["output_dtype"] == DataType.BF16 else 0

        return PrimPooling(
            deps=deps,
            x_in_addr=x_in_addr,
            bias_value_or_addr=bias_value_or_addr,
            out_addr=y_out_addr,
            para_addr=para_addr,
            batch_size=batch_size - 1,
            x_in_h=x_in_h - 1,
            x_in_w=x_in_w - 1,
            c_in_32B=c_in_32b - 1,
            kernel_h=kernel_h - 1,
            kernel_w=kernel_w - 1,
            scaler=self.attrs["scaler"],
            scaler_mode=self.attrs["scaler_mode"],
            max_or_min=self.pool_type_number,
            y_type=y_type,
            bias_mode=self.attrs["bias_mode"],
            stride_h=stride_h - 1,
            stride_w=stride_w - 1,
        )

    def _resolve_bias_value_or_node(self, inputs: Dict[int, Data]) -> int:
        bias_mode = self.attrs["bias_mode"]
        if bias_mode == 0:
            return 0
        if bias_mode == 1:
            return self.attrs["bias_scalar"]
        if bias_mode == 2:
            bias_node = inputs.get(1)
            if bias_node is None or bias_node.memref is None:
                raise ValueError("MaxPooling bias_node on port 1.")
            return bias_node
        raise ValueError(f"Unsupported bias_mode {bias_mode} for MaxPooling.")