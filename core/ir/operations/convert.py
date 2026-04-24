from math import ceil
from typing import Dict, List, Optional, Tuple

from myhdl import intbv

from core.ir.operation import Operation
from core.ir.data import Data, DataType, elements_to_32b_cell, ViewData
from core.ir.prims.convert import PrimConvert


class Convert(Operation):
    """Data type conversion backed by :class:`PrimConvert`.

    Ports:
        inputs:
            - 0: source tensor (2D, INT8/BF16/SPIKE)
        outputs:
            - 0: converted tensor (2D, INT8/BF16/SPIKE)

    Attributes:
        attrs (Dict[str, Any]): Operation parameters dictionary. Expected keys:
            - 'output_dtype' (DataType): Desired output dtype (INT8/BF16/SPIKE).

    Methods:
        infer(inputs): Validate input metadata and infer output shape/dtype.
        gen_prim(inputs, outputs, deps): Build the hardware primitive instruction.
        _compute_output_cells(input_cells, in_dtype, out_dtype): Derive 32B cell count after conversion.
        _dtype_to_hw_flag(dtype): Map framework dtype to hardware flag encoding.
    """

    def __init__(self, name: str, attrs: Optional[Dict[str, object]] = None) -> None:
        super().__init__(name, attrs)

        required_attrs = ["input_dtype", "output_dtype"]
        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for Convert operation.")

        self.supported_input_dtype = (DataType.INT8, DataType.BF16, DataType.SPIKE)
        self.supported_output_dtype = (DataType.INT8, DataType.BF16, DataType.SPIKE)

        if self.attrs["input_dtype"] not in self.supported_input_dtype:
            raise ValueError(
                f"Unsupported input_dtype {self.attrs['input_dtype']}. Supported: {self.supported_input_dtype}."
            )

        if self.attrs["output_dtype"] not in self.supported_output_dtype:
            raise ValueError(
                f"Unsupported output_dtype {self.attrs['output_dtype']}. Supported: {self.supported_output_dtype}."
            )
        
        if self.attrs["input_dtype"] == self.attrs["output_dtype"]:
            raise ValueError("Input and output dtypes for Convert operation must differ.")

        self.primitive = True # Marking as a primitive operation

    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        if 0 not in inputs:
            raise ValueError("Convert requires an input tensor on port 0.")

        input_data = inputs[0]
        if len(input_data.shape) != 2:
            raise ValueError("Convert expects 2D input tensors on port 0.")

        if input_data.dtype != self.attrs["input_dtype"]:
            raise ValueError(f"input data dtype {input_data.dtype} must match Convert input dtype {self.attrs['input_dtype']}.")

        output_dtype = self.attrs["output_dtype"]

        # vector_num_in = input_data.shape[0]
        # vector_len_in_cells = elements_to_32b_cell(input_data.shape[1], input_data.dtype)

        # vector_num_out = vector_num_in
        # vector_len_out_cells = self._compute_output_cells(vector_len_in_cells, input_data.dtype, output_dtype)

        # output_shape = (vector_num_out, vector_len_out_cells)

        output_shape = input_data.shape
        return [(output_shape, output_dtype)]

    def para_node(self, inputs: Dict[int, Data], outputs: Dict[int, Data]) -> Optional[Data]:
        # For Convert operations, we typically don't need many parameters
        return None

    def para_connection(self) -> bool:
        '''
        True: double connection (input and output)
        False: single connection (only input)
        '''
        return False

    def to_prim(self):
        """Convert to a primitive operation representation if needed."""
        # Convert is already implemented as a primitive operation, return None
        return None

    def gen_prim(
        self,
        inputs: Dict[int, Data],
        outputs: Dict[int, Data],
        deps: intbv = intbv(0)[8:],
    ) -> intbv:
        input_data = inputs[0]
        output_data = outputs[0]

        primitive = self._build_primitive(input_data, output_data, deps)
        return primitive.PIC

    def build_prim(
        self,
        inputs: Dict[int, Data],
        outputs: Dict[int, Data],
        deps: intbv = intbv(0)[8:],
    ) -> PrimConvert:
        input_data = inputs[0]
        output_data = outputs[0]
        return self._build_primitive(input_data, output_data, deps)

    def _build_primitive(
        self,
        input_data: Data,
        output_data: Data,
        deps: intbv,
    ) -> PrimConvert:
        output_dtype: DataType = self.attrs["output_dtype"]
        if output_data.dtype is not None and output_data.dtype != output_dtype:
            raise ValueError(
                f"Output data dtype {output_data.dtype} mismatches Convert output dtype {output_dtype}."
            )

        vector_num = input_data.shape[0]
        input_vector_len_cells = elements_to_32b_cell(input_data.shape[1], input_data.dtype)
        x_addr = input_data.inferred_memref.addr if isinstance(input_data, ViewData) else input_data.memref.addr

        primitive = PrimConvert(
            deps=int(deps),
            x_in_addr=x_addr,
            y_out_addr=output_data.memref.addr,
            vector_num=vector_num - 1,
            vector_len_in_32B=input_vector_len_cells - 1,
            x_in_type=self._dtype_to_hw_flag(input_data.dtype),
            y_out_type=self._dtype_to_hw_flag(output_dtype),
        )
        return primitive

    @staticmethod
    def _dtype_to_hw_flag(dtype: DataType) -> int:
        mapping = {
            DataType.INT8: 0,
            DataType.BF16: 1,
            DataType.SPIKE: 2,
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported dtype {dtype} for Convert primitive.")
        return mapping[dtype]

    @staticmethod
    def _compute_output_cells(input_cells: int, in_dtype: DataType, out_dtype: DataType) -> int:
        if input_cells <= 0:
            raise ValueError("Input vector length in cells must be positive.")

        if in_dtype == out_dtype:
            return input_cells

        if in_dtype == DataType.INT8 and out_dtype == DataType.BF16:
            return input_cells * 2
        if in_dtype == DataType.BF16 and out_dtype == DataType.INT8:
            return int(ceil(input_cells / 2))

        if in_dtype == DataType.INT8 and out_dtype == DataType.SPIKE:
            return int(ceil(input_cells / 8))
        if in_dtype == DataType.BF16 and out_dtype == DataType.SPIKE:
            return int(ceil(input_cells / 16))

        if in_dtype == DataType.SPIKE and out_dtype == DataType.INT8:
            return input_cells * 8
        if in_dtype == DataType.SPIKE and out_dtype == DataType.BF16:
            return input_cells * 16

        raise ValueError(f"Unsupported conversion from {in_dtype} to {out_dtype}.")
