from typing import Dict, List, Optional, Tuple

from myhdl import intbv

from core.ir.operation import Operation
from core.ir.data import Data, DataType, MemBlock, elements_to_32b_cell
from core.ir.prims.maxpooling import PrimPooling


class Max(Operation):
	"""Row-wise max reduction implemented with a single MaxPooling primitive.
	
	Ports:
        inputs:
            - 0: input tensor (2D (vector_num, vector_len), BF16 only)
            - 1: para
        outputs:
            - 0: output tensor (2D (1, vector_len), BF16 or INT8)

    Attributes:
        name (str): Human-readable identifier for the operation.
        required attrs (Dict[str, Any]): Operation parameters dictionary. Expected keys:
            - 'output_dtype' (DataType): Desired output tensor type.(BF16 or INT8)
	"""

	def __init__(self, name: str, attrs: Optional[Dict[str, object]] = None) -> None:
		super().__init__(name, attrs)

		if self.attrs["output_dtype"] not in (DataType.BF16, DataType.INT8):
			raise ValueError("'output_dtype' must be DataType.BF16 or DataType.INT8.")

		self.primitive = True

	def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
		if 0 not in inputs:
			raise ValueError("Max expects input tensor on port 0.")

		input_data = inputs[0]
		if input_data.shape is None or len(input_data.shape) != 2:
			raise ValueError("Max input tensor must have shape (vector_num, vector_len).")

		if input_data.dtype != DataType.BF16:
			raise ValueError("Max currently supports BF16 input tensors only.")

		vector_num, vector_len = input_data.shape
		if vector_num <= 0 or vector_len <= 0:
			raise ValueError("Input dimensions must be positive integers for Max operation.")

		output_shape = (1, vector_len)
		return [(output_shape, self.attrs["output_dtype"])]

	def para_node(self, inputs: Dict[int, Data], outputs: Dict[int, Data]) -> Optional[Data]:
		input_data = inputs[0]

		prim_max = self._build_primitive(
			input_data=input_data,
			x_in_addr=0,
			y_out_addr=0,
			para_addr=0,
			deps=intbv(0)[8:]
		)

		para_code = MemBlock(length=1, payload=[prim_max.para])
		return Data(name=f"{self.name}.params", memref=para_code)

	def para_connection(self) -> bool:
		return False

	def gen_prim(
		self,
		inputs: Dict[int, Data],
		outputs: Dict[int, Data],
		deps: intbv = intbv(0)[8:],
	) -> intbv:
		input_data = inputs[0]
		output_data = outputs[0]
		para_data = inputs[1]

		prim_max = self._build_primitive(
			input_data=input_data,
			x_in_addr=input_data.memref.addr,
			y_out_addr=output_data.memref.addr,
			para_addr=para_data.memref.addr,
			deps=deps,
		)

		return prim_max.PIC

	def _build_primitive(
		self,
		input_data: Data,
		x_in_addr: int,
		y_out_addr: int,
		para_addr: int,
		deps: intbv,
	) -> PrimPooling:
		batch_size, x_in_h, x_in_w, channels = self._expand_to_pooling_shape(input_data.shape)
		c_in_32b = elements_to_32b_cell(channels, input_data.dtype)

		kernel_h = 1
		kernel_w = x_in_w
		stride_h = 1
		stride_w = x_in_w

		y_type = 1 if self.attrs["output_dtype"] == DataType.BF16 else 0

		return PrimPooling(
			deps=deps,
			x_in_addr=x_in_addr,
			bias_value_or_addr=0,
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
			max_or_min=0,
			y_type=y_type,
			bias_mode=0,
			stride_h=stride_h - 1,
			stride_w=stride_w - 1,
		)

	def _expand_to_pooling_shape(self, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
		vector_num, vector_len = shape
		return (1, 1, vector_num, vector_len)
