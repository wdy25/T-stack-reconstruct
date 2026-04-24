import math
from typing import Dict, List, Optional, Tuple

from myhdl import intbv

from core.ir.operation import Operation
from core.ir.data import Data, DataType, elements_to_32b_cell
from core.ir.prims.add import PrimAdd


class Add(Operation):
    """Element-wise add/subtract operation with optional broadcast and scalar modes.
    Ports:
        inputs:
            - 0: primary input tensor (shape>=2D, BF16 only)
            - 1: secondary input tensor
                - when bc_mode == 0: shape must match primary input shape
                - when bc_mode == 1: last dimension must match primary input last dimension, others must be 1. len(shape) must match primary input.
                - when bc_mode == 2: scalar add, secondary input should be None
        outputs:
            - 0: output tensor

    Attributes:
        attrs (Dict[str, Any]): Expected keys
            - 'output_dtype' (DataType): Output tensor dtype.
            - 'bc_mode' (int): 0 -> no broadcast, 1 -> vector broadcast, 2 -> scalar add.
            - 'add_or_sub' (int): 0 -> add, 1 -> subtract.
            - 'scalar' (float): scalar literal used when bc_mode == 2.

    Methods:
        infer(inputs: List[Data], ports: List[int]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
            Infers the output shape and dtype based on input metadata and operation attributes.
        para_node() -> MemBlock:
            Create a parameter node for add operation.
    """

    def __init__(self, name: str, attrs: Optional[Dict[str, object]] = None) -> None:
        super().__init__(name, attrs)
        required_attrs = [
            "output_dtype",
            "bc_mode",
            "add_or_sub",
        ]
        optional_attrs = [
            "scalar",  # required if bc_mode == 2
        ]
        # Set default values for some attributes
        self.attrs.setdefault("bc_mode", 0)
        self.attrs.setdefault("add_or_sub", 0)
        self.attrs.setdefault("scalar", 0.0)

        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for Add operation.")

        supported_output_dtype = (DataType.BF16, DataType.INT8)
        if self.attrs["output_dtype"] not in supported_output_dtype:
            raise ValueError(
                f"Unsupported output_dtype {self.attrs['output_dtype']}. Supported: {supported_output_dtype}."
            )

        if self.attrs["bc_mode"] not in (0, 1, 2):
            raise ValueError("'bc_mode' must be 0, 1, or 2.")
        if self.attrs["add_or_sub"] not in (0, 1):
            raise ValueError("'add_or_sub' must be 0 or 1.")

        scalar_value = self.attrs["scalar"]
        if not isinstance(scalar_value, (int, float)):
            raise ValueError("'scalar' must be numeric.")
        self.attrs["scalar"] = float(scalar_value)

        self.primitive = True # Marking as a primitive operation

    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        if 0 not in inputs:
            raise ValueError("AddNew requires an input tensor on port 0.")

        input_data_a = inputs[0]
        input_data_b = inputs.get(1)
        bc_mode = self.attrs["bc_mode"]

        if len(input_data_a.shape) < 2:
            raise ValueError("Primary input shape must be at least 2D on port 0.")
        
        supported_input_dtype = (DataType.BF16,)
        if input_data_a.dtype not in supported_input_dtype:
            raise ValueError(f"Unsupported primary dtype {input_data_a.dtype}.")

        if bc_mode in (0, 1):
            if input_data_b is None:
                raise ValueError("Secondary input is required when bc_mode is 0 or 1.")
            self._validate_secondary(input_data_a, input_data_b, bc_mode)
        else:
            if input_data_b is not None:
                raise ValueError("Secondary input must be None when bc_mode is 2.")
            
        # vector_num_in = input_data_a.shape[0]
        # vector_len_in = elements_to_32b_cell(input_data_a.shape[1], input_data_a.dtype)
        
        # vector_num_out = vector_num_in
        # vector_len_out = vector_len_in
        # if self.attrs["output_dtype"] == DataType.BF16:
        #     vector_len_out = vector_len_in
        # elif self.attrs["output_dtype"] == DataType.INT8:
        #     vector_len_out = int(ceil(vector_len_in / 2))

        # output_shape = (vector_num_out, vector_len_out)
        output_shape = input_data_a.shape

        return [(output_shape, self.attrs["output_dtype"])]

    def para_node(self, inputs: Dict[int, Data], outputs: Dict[int, Data]) -> Optional[Data]:
        """Create a parameter node for add operation.
        
        For element-wise addition, minimal parameters are needed.
        This might include quantization parameters or scaling factors
        for different data types.
        """
        # For basic addition operations, we typically don't need many parameters
        # This could be extended to include scaling factors, bias terms, etc.
        return None
    
    def para_connection(self) -> bool:
        '''
        True: double connection (input and output)
        False: single connection (only input)
        '''
        return False

    def to_prim(self):
        """Convert to a primitive operation representation if needed."""
        # Add is already implemented as a primitive operation, return None
        return None

    def gen_prim(
        self,
        inputs: Dict[int, Data],
        outputs: Dict[int, Data],
        deps: intbv = intbv(0)[8:],
    ) -> intbv:
        
        input_data_a = inputs[0]
        input_data_b = inputs.get(1)
        output_data = outputs[0]

        primitive = self._build_primitive(input_data_a, input_data_b, output_data, deps)
        return primitive.PIC

    def build_prim(self,
        inputs: Dict[int, Data],
        outputs: Dict[int, Data],
        deps: intbv = intbv(0)[8:],
    ):
        input_data_a = inputs[0]
        input_data_b = inputs.get(1)
        output_data = outputs[0]

        primitive = self._build_primitive(input_data_a, input_data_b, output_data, deps)
        return primitive

    def _build_primitive(
        self,
        input_data_a: Data,
        input_data_b: Optional[Data],
        output_data: Data,
        deps: intbv,
    ) -> PrimAdd:
        bc_mode = self.attrs["bc_mode"]

        if bc_mode in (0, 1):
            assert input_data_b is not None
            if input_data_b.dtype != input_data_a.dtype:
                raise ValueError("Primary and secondary inputs must share the same dtype.")

        vector_num = math.prod(input_data_a.shape[:-1])
        vector_len = elements_to_32b_cell(input_data_a.shape[-1], input_data_a.dtype)
        y_out_type = 1 if self.attrs["output_dtype"] == DataType.BF16 else 0
        x_in_2_addr = input_data_b.memref.addr if input_data_b else 0

        return PrimAdd(
            deps=int(deps),
            x_in_1_addr=input_data_a.memref.addr,
            x_in_2_addr=x_in_2_addr,
            y_out_addr=output_data.memref.addr,
            vector_num=vector_num - 1,
            vector_len_in_32B=vector_len -1,
            scalar=self.attrs["scalar"],
            y_out_type=y_out_type,
            bc_mode=self.attrs["bc_mode"],
            add_or_sub=self.attrs["add_or_sub"],
        )

    def _validate_secondary(self, input_data_a: Data, input_data_b: Data, bc_mode: int) -> None:

        if bc_mode == 0:
            if input_data_b.shape != input_data_a.shape:
                raise ValueError(
                    f"Secondary shape {input_data_b.shape} must match primary shape {input_data_a.shape} when bc_mode is 0."
                )
        elif bc_mode == 1:
            expected_last_dim = input_data_a.shape[-1]
            if input_data_b.shape[-1] != expected_last_dim:
                raise ValueError(
                    f"Secondary tensor last dimension {input_data_b.shape[-1]} must match primary last dimension {expected_last_dim} when bc_mode is 1."
                )

            if len(input_data_b.shape) == len(input_data_a.shape):
                if any(dim != 1 for dim in input_data_b.shape[:-1]):
                    raise ValueError(
                        "Secondary tensor must broadcast across leading dimensions (all ones) when bc_mode is 1."
                    )
            else:
                raise ValueError(
                    "Secondary tensor must match primary rank with leading ones when bc_mode is 1, for example (1, 1, N) for primary (B, C, N)."
                )