from math import ceil
from typing import Any, Dict, List, Optional, Tuple

from myhdl import intbv

from core.ir.operation import Operation
from core.ir.data import Data, DataType, MemBlock, elements_to_32b_cell
from core.ir.prims.meanpooling import PrimMeanPooling


class MeanPooling(Operation):
    """Meanpooling operation backed by :class:`PrimMeanPooling`.

    Ports: 
        inputs:
            - 0: input tensor (4D (N, H_in, W_in, C_in), BF16 only)
            - 1: para
        outputs:
            - 0: output tensor (4D (N, H_out, W_out, C_in), BF16 or INT8)

    Attributes:
        name (str): Human-readable identifier for the operation.
        attrs (Dict[str, Any]): Operation parameters dictionary. Expected keys:
            - 'kernel_size' (Tuple[int, int]): Size of the pooling kernel.
            - 'stride' (Tuple[int, int]): Stride applied during pooling.
            - 'scaler' (int): Integer scaling factor applied when `scaler_mode` enables it.
            - 'scaler_mode' (int): Hardware scaling mode select bit (see table below).
            - 'output_dtype' (DataType): Desired output tensor type.

    Methods:
        infer: Derive output tensor metadata from the provided input tensor.
        para_node: Create the hardware para block required by the primitive.
        para_connection: Indicate whether para generation needs output linkage.
        gen_prim: Assemble the final PIC instruction for hardware execution.

    Attribute bit usage (PrimMeanPooling PIC field reference):
        - scaler_mode: width=1, bits=[167]; 0 disables scaling, 1 applies the provided `scaler` value.
    """

    def __init__(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, attrs)
        required_attrs = [
            "kernel_size",
            "stride",
            "scaler",
            "scaler_mode",
            "output_dtype",
        ]
        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for MeanPooling operation.")

        if not isinstance(self.attrs["kernel_size"], tuple) or len(self.attrs["kernel_size"]) != 2:
            raise ValueError("'kernel_size' must be a tuple of two integers.")
        if not isinstance(self.attrs["stride"], tuple) or len(self.attrs["stride"]) != 2:
            raise ValueError("'stride' must be a tuple of two integers.")
        if not all(isinstance(x, int) for x in self.attrs["kernel_size"]):
            raise ValueError("'kernel_size' entries must be integers.")
        if not all(isinstance(x, int) for x in self.attrs["stride"]):
            raise ValueError("'stride' entries must be integers.")
        if not isinstance(self.attrs["scaler"], int):
            raise ValueError("'scaler' must be an integer.")

        supported_outputs = (DataType.BF16, DataType.INT8)
        if self.attrs["output_dtype"] not in supported_outputs:
            raise ValueError(
                f"Unsupported output_dtype {self.attrs['output_dtype']}. Supported: {supported_outputs}."
            )

        if self.attrs["scaler_mode"] not in (0, 1):
            raise ValueError("'scaler_mode' must be 0 or 1.")

        self.primitive = True  # Marking as a primitive operation

    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        # MeanPooling requires input on port 0
        input_data = inputs[0]
        if input_data.shape is None or len(input_data.shape) != 4:
            raise ValueError("Input data must have shape (N, H_in, W_in, C_in).")

        if input_data.dtype != DataType.BF16:
            raise ValueError("MeanPooling currently supports only BF16 input tensors.")

        kernel_h, kernel_w = self.attrs["kernel_size"]
        stride_h, stride_w = self.attrs["stride"]

        N, H_in, W_in, C_in = input_data.shape
        if H_in < kernel_h or W_in < kernel_w:
            raise ValueError("Pooling kernel larger than input spatial dimensions.")
        if stride_h <= 0 or stride_w <= 0:
            raise ValueError("Stride values must be positive integers.")

        H_out = ((H_in - kernel_h) // stride_h) + 1
        W_out = ((W_in - kernel_w) // stride_w) + 1

        # if self.attrs["output_dtype"] == DataType.BF16:
        #     C_out = C_in
        # else:
        #     C_out = int(ceil(C_in / 2))

        output_shape = (N, H_out, W_out, C_in)
        return [(output_shape, self.attrs["output_dtype"])]

    def para_node(self, inputs: Dict[int, Data], outputs: Dict[int, Data]) -> Optional[Data]:
        # MeanPooling gen_prim requires input and output on port 0
        input_data = inputs[0]

        prim_meanpooling = self._build_primitive(
            input_data=input_data,
            x_in_addr=0,
            y_out_addr=0,
            para_addr=0,
            deps=intbv(0)[8:],
        )

        para_code = MemBlock(length=1, payload=[prim_meanpooling.para])
        para_data = Data(name=f"{self.name}.params", memref=para_code)
        return para_data

    def para_connection(self) -> bool:
        """
        True: double connection (input and output)
        False: single connection (only input)
        """
        return False

    def gen_prim(
        self,
        inputs: Dict[int, Data],
        outputs: Dict[int, Data],
        deps: intbv = intbv(0)[8:],
    ) -> intbv:
        # MeanPooling gen_prim requires input and output on port 0
        input_data = inputs[0]
        output_data = outputs[0]

        # MeanPooling gen_prim requires para on last port(port 1)
        para_data = inputs[1]

        x_in_addr = input_data.memref.addr
        y_out_addr = output_data.memref.addr
        para_addr = para_data.memref.addr

        assert x_in_addr is not None and y_out_addr is not None and para_addr is not None, \
            "Input, output, and para data must have valid memory addresses."
        assert x_in_addr >= 0x0 and y_out_addr >= 0x0 and para_addr >= 0x0, \
            "Input, output, and para data addresses must be non-negative."

        prim_meanpooling = self._build_primitive(
            input_data=input_data,
            x_in_addr=x_in_addr,
            y_out_addr=y_out_addr,
            para_addr=para_addr,
            deps=deps,
        )

        return prim_meanpooling.PIC
    
    def build_prim(
        self,
        inputs: Dict[int, Data],
        outputs: Dict[int, Data],
        deps: intbv = intbv(0)[8:],
    ) -> PrimMeanPooling:
        # MeanPooling gen_prim requires input and output on port 0
        input_data = inputs[0]
        output_data = outputs[0]

        # MeanPooling gen_prim requires para on last port(port 1)
        para_data = inputs[1]

        x_in_addr = input_data.memref.addr
        y_out_addr = output_data.memref.addr
        para_addr = para_data.memref.addr

        assert x_in_addr is not None and y_out_addr is not None and para_addr is not None, \
            "Input, output, and para data must have valid memory addresses."
        assert x_in_addr >= 0x0 and y_out_addr >= 0x0 and para_addr >= 0x0, \
            "Input, output, and para data addresses must be non-negative."

        return self._build_primitive(
            input_data=input_data,
            x_in_addr=x_in_addr,
            y_out_addr=y_out_addr,
            para_addr=para_addr,
            deps=deps,
        )

    def _build_primitive(
        self,
        input_data: Data,
        x_in_addr: int,
        y_out_addr: int,
        para_addr: int,
        deps: intbv,
    ) -> PrimMeanPooling:
        batch_size = input_data.shape[0]
        x_in_h = input_data.shape[1]
        x_in_w = input_data.shape[2]
        c_in_32b = elements_to_32b_cell(input_data.shape[3], input_data.dtype)

        kernel_h, kernel_w = self.attrs["kernel_size"]
        stride_h, stride_w = self.attrs["stride"]

        y_type = 1 if self.attrs["output_dtype"] == DataType.BF16 else 0

        return PrimMeanPooling(
            deps=deps,
            x_in_addr=x_in_addr,
            y_out_addr=y_out_addr,
            para_addr=para_addr,
            batch_size=batch_size - 1,
            x_in_h=x_in_h - 1,
            x_in_w=x_in_w - 1,
            c_in_32B=c_in_32b - 1,
            kernel_h=kernel_h - 1,
            kernel_w=kernel_w - 1,
            scaler=self.attrs["scaler"],
            scaler_mode=self.attrs["scaler_mode"],
            y_out_type=y_type,
            stride_h=stride_h - 1,
            stride_w=stride_w - 1,
        )
