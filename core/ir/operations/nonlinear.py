import math
from typing import Any, Dict, List, Optional, Tuple

from myhdl import intbv

from core.ir.operation import Operation
from core.ir.data import Data, DataType, elements_to_32b_cell
from core.ir.prims.nonlinear import PrimNonlinear


class Nonlinear(Operation):
    """Non-linear elementary function backed by :class:`PrimNonlinear`.

    Ports:
        inputs:
            - 0: source tensor (shape>=2D, BF16 only)
        outputs:
            - 0: transformed tensor (the same shape as inputs[0], BF16 or INT8)

    Attributes:
        required attrs (Dict[str, Any]): Operation parameters dictionary. Expected keys:
            - 'output_dtype' (DataType): Output tensor dtype (BF16 or INT8).
            - 'function' (str): Non-linear function name supported by hardware.
                - Supported functions: "sqrt", "sin", "cos", "exp".
    """

    def __init__(self, name: str, attrs: Optional[Dict[str, object]] = None) -> None:
        super().__init__(name, attrs)

        required_attrs = ["output_dtype", "function"]
        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for Nonlinear operation.")

        supported_output_dtype = (DataType.BF16, DataType.INT8)
        if self.attrs["output_dtype"] not in supported_output_dtype:
            raise ValueError(
                f"Unsupported output_dtype {self.attrs['output_dtype']}. Supported: {supported_output_dtype}."
            )

        func_name = str(self.attrs["function"]).lower()
        supported_functions = PrimNonlinear.get_supported_functions()
        if func_name not in supported_functions:
            raise ValueError(f"Unsupported function '{func_name}'. Supported: {supported_functions}.")

        self.attrs["function"] = func_name
        self._func_code = PrimNonlinear.get_func_value(func_name)

        self.primitive = True  # Marking as a primitive operation

    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        if 0 not in inputs:
            raise ValueError("Nonlinear requires an input tensor on port 0.")

        input_data = inputs[0]
        if len(input_data.shape) < 2:
            raise ValueError("Nonlinear expects shape>=2D input tensors on port 0.")

        if input_data.dtype != DataType.BF16:
            raise ValueError("Nonlinear supports only BF16 inputs.")

        # vector_num_in = input_data.shape[0]
        # vector_len_in = elements_to_32b_cell(input_data.shape[1], input_data.dtype)

        # output_dtype = self.attrs["output_dtype"]
        # if output_dtype == DataType.BF16:
        #     vector_len_out = vector_len_in
        # elif output_dtype == DataType.INT8:
        #     vector_len_out = int(ceil(vector_len_in / 2))
        # else:
        #     raise ValueError(f"Unsupported output dtype {output_dtype} for Nonlinear operation.")
        # output_shape = (vector_num_in, vector_len_out)
        
        return [(input_data.shape, self.attrs["output_dtype"])]

    def para_node(self, inputs: Dict[int, Data], outputs: Dict[int, Data]) -> Optional[Data]:
        # For Nonlinear operations, we typically don't need many parameters
        return None

    def para_connection(self) -> bool:
        '''
        True: double connection (input and output)
        False: single connection (only input)
        '''
        return False

    def to_prim(self):
        """Convert to a primitive operation representation if needed."""
        # Nonlinear is already implemented as a primitive operation, return None
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
    ) -> PrimNonlinear:
        input_data = inputs[0]
        output_data = outputs[0]
        return self._build_primitive(input_data, output_data, deps)

    def _build_primitive(
        self,
        input_data: Data,
        output_data: Data,
        deps: intbv,
    ) -> PrimNonlinear:

        vector_num = math.prod(input_data.shape[:-1])
        vector_len_in_cells = elements_to_32b_cell(input_data.shape[-1], input_data.dtype)
        y_out_type = 1 if self.attrs["output_dtype"] == DataType.BF16 else 0

        return PrimNonlinear(
            deps=int(deps),
            x_in_addr=input_data.memref.addr,
            func=self._func_code,
            y_out_addr=output_data.memref.addr,
            vector_num=vector_num -1,
            vector_len_32B=vector_len_in_cells -1,
            out_type=y_out_type,
        )
