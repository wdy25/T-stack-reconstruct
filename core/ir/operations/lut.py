from math import ceil
from typing import Dict, List, Optional, Tuple
import torch

from myhdl import intbv

from core.ir.operation import Operation
from core.ir.data import Data, DataType, elements_to_32b_cell
from core.ir.prims.lut import PrimMeanLUT


class LUT(Operation):
    """Lookup-table based non-linear transform backed by :class:`PrimMeanLUT`.

    Ports:
        inputs:
            - 0: source tensor (2D, INT8 or BF16)
            - 1: LUT table tensor (INT8 or BF16 matching output)
        outputs:
            - 0: looked-up tensor

    Attributes:
        name (str): Human-readable identifier for the operation.
        attrs (Dict[str, Any]): Operation parameters dictionary. Expected keys:
            - 'input_dtype' (DataType): Input tensor dtype (INT8 or BF16).
            - 'output_dtype' (DataType): Output tensor dtype (INT8 or BF16).
    """

    def __init__(self, name: str, attrs: Optional[Dict[str, object]] = None) -> None:
        super().__init__(name, attrs)
        required_attrs = ["input_dtype", "output_dtype"]
        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for LUT operation.")

        self.supported_input_dtype = (DataType.BF16, DataType.INT8)
        if self.attrs["input_dtype"] not in self.supported_input_dtype:
            raise ValueError(
                f"Unsupported input_dtype {self.attrs['input_dtype']}. Supported: {self.supported_input_dtype}."
            )

        self.supported_output_dtype = (DataType.BF16, DataType.INT8)
        if self.attrs["output_dtype"] not in self.supported_output_dtype:
            raise ValueError(
                f"Unsupported output_dtype {self.attrs['output_dtype']}. Supported: {self.supported_output_dtype}."
            )

        self.primitive = True # Marking as a primitive operation

    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        if 0 not in inputs:
            raise ValueError("LUT requires an input tensor on port 0.")
        if 1 not in inputs:
            raise ValueError("LUT requires an LUT tensor on port 1.")

        input_data = inputs[0]
        if len(input_data.shape) != 2:
            raise ValueError("LUT expects 2D input tensors on port 0.")

        input_dtype = input_data.dtype
        if input_dtype not in self.supported_input_dtype:
            raise ValueError(f"Unsupported input dtype {input_dtype}. Supported: {self.supported_input_dtype}.")

        lut_data = inputs[1]
        if lut_data.dtype != self.attrs["output_dtype"]:
            raise ValueError(
                f"LUT tensor dtype {lut_data.dtype} must match output dtype {self.attrs['output_dtype']}."
            )
        
        lut_data_size = self._get_total_lut_size(lut_data)
        if input_dtype == DataType.INT8:
            # INT8 input: 256 possible values (0-255)
            if lut_data_size != 256:
                raise ValueError("LUT tensor size must be 256 for INT8 input tensors.")
        elif input_dtype == DataType.BF16:
            # BF16 input: 65536 possible values (0-65535)
            if lut_data_size != 65536:
                raise ValueError("LUT tensor size must be 65536 for BF16 input tensors.")
        
        # vector_num_in = input_data.shape[0]
        # elements_in_per_vector = input_data.shape[1]
        # vector_len_in = elements_to_32b_cell(elements_in_per_vector, input_dtype)
        
        # vector_num_out = vector_num_in
        # if self.attrs["output_dtype"] == DataType.BF16:
        #     vector_len_out = vector_len_in
        # elif self.attrs["output_dtype"] == DataType.INT8:
        #     vector_len_out = ceil(vector_len_in / 2)
        
        output_dtype = self.attrs["output_dtype"]

        output_shape = input_data.shape
        return [(output_shape, output_dtype)]

    def para_node(self, inputs: Dict[int, Data], outputs: Dict[int, Data]) -> Optional[Data]:
        # For LUT operations, we typically don't need many parameters
        return None

    def para_connection(self) -> bool:
        '''
        True: double connection (input and output)
        False: single connection (only input)
        '''
        return False

    def to_prim(self):
        """Convert to a primitive operation representation if needed."""
        # LUT is already implemented as a primitive operation, return None
        return None

    def gen_prim(
        self,
        inputs: Dict[int, Data],
        outputs: Dict[int, Data],
        deps: intbv = intbv(0)[8:],
    ) -> intbv:
        input_data = inputs[0]
        lut_data = inputs[1]

        output_data = outputs[0]
        primitive = self._build_primitive(input_data, lut_data, output_data, deps)
        return primitive.PIC

    def build_prim(
        self,
        inputs: Dict[int, Data],
        outputs: Dict[int, Data],
        deps: intbv = intbv(0)[8:],
    ) -> PrimMeanLUT:
        input_data = inputs[0]
        lut_data = inputs[1]
        output_data = outputs[0]
        return self._build_primitive(input_data, lut_data, output_data, deps)

    def _build_primitive(
        self,
        input_data: Data,
        lut_data: Data,
        output_data: Data,
        deps: intbv,
    ) -> PrimMeanLUT:
        vector_num = input_data.shape[0]
        vector_len_in_cells = elements_to_32b_cell(input_data.shape[1], input_data.dtype)

        x_in_type = self._dtype_to_hw_flag(input_data.dtype)
        y_out_type = self._dtype_to_hw_flag(self.attrs["output_dtype"])

        return PrimMeanLUT(
            deps=int(deps),
            x_in_addr=input_data.memref.addr,
            lut_in_addr=lut_data.memref.addr,
            y_out_addr=output_data.memref.addr,
            vector_num=vector_num - 1,
            vector_len_in_32B=vector_len_in_cells - 1,
            x_in_type=x_in_type,
            y_out_type=y_out_type,
        )

    @staticmethod
    def _dtype_to_hw_flag(dtype: DataType) -> int:
        if dtype == DataType.INT8:
            return 0
        if dtype == DataType.BF16:
            return 1
        raise ValueError(f"Unsupported dtype for LUT primitive: {dtype}.")
    
    @staticmethod
    def _get_total_lut_size(lut_data: Data):
        """compute total LUT size in element dimension"""
        return lut_data.shape[0] * lut_data.shape[1]
    
    @staticmethod
    def generate_lut_table(x_in_type: str = "INT8", y_out_type: str = "BF16", function: str = "sigmoid"):
        """根据输入输出类型生成LUT查找表，支持多种激活函数
        
        Args:
            x_in_type: 输入数据类型，"INT8" 或 "BF16"
            y_out_type: 输出数据类型，"INT8" 或 "BF16"
            function: 激活函数类型，支持 "sigmoid", "softmax", "gelu", "tanh", "relu" 等
            
        Returns:
            展平的LUT查找表 tensor
        """
        # 确定输入大小
        if x_in_type == 'INT8':
            # INT8输入：256个可能的值 (0-255)
            lut_input_size = 256
            # 遍历所有可能的输入值，将8bit索引转换为int8值
            input_values = torch.zeros(lut_input_size, dtype=torch.int8)
            for i in range(lut_input_size):
                # 将i的8bit表示转换为int8值：0-127映射到0-127，128-255映射到-128到-1
                if i < 128:
                    value_int8 = i
                else:
                    value_int8 = i - 256
                input_values[i] = value_int8
        else:  # BF16
            # BF16输入：65536个可能的值 (0-65535)
            lut_input_size = 65536
            # 遍历所有可能的输入值，将16bit索引转换为bf16值
            input_values = torch.zeros(lut_input_size, dtype=torch.bfloat16)
            for i in range(lut_input_size):
                # 将i的16bit表示为uint16，再通过view转换为位值对应的bf16
                value_uint16 = torch.tensor(i, dtype=torch.uint16)
                # 使用torch将位模式转换为实际的bf16值
                value_bf16 = value_uint16.view(torch.bfloat16)
                input_values[i] = value_bf16

        # input_values统一转化为float32进行计算
        input_values_FP32 = input_values.to(torch.float32)
        
        # 根据指定的函数计算输出值
        function_lower = function.lower()
        if function_lower == "sigmoid":
            output_values_FP32 = torch.sigmoid(input_values_FP32)
        elif function_lower == "tanh":
            output_values_FP32 = torch.tanh(input_values_FP32)
        elif function_lower == "relu":
            output_values_FP32 = torch.relu(input_values_FP32)
        elif function_lower == "gelu":
            output_values_FP32 = torch.nn.functional.gelu(input_values_FP32)
        else:
            raise ValueError(f"不支持的激活函数: {function}. 支持的函数: sigmoid, tanh, relu, gelu")

        # 根据输出类型进行转换
        if y_out_type == 'INT8':
            # 输出INT8：将float输出值转换为int8
            # 直接clamp到int8范围并四舍五入
            lut_values = output_values_FP32.round().clamp(-128, 127).to(torch.int8)
        else:  # BF16
            # 输出BF16：直接转换为bfloat16
            lut_values = output_values_FP32.to(torch.bfloat16)

        # 重新组织LUT表，按照cell和position的方式存储
        # 这样在查找时，high_bits确定cell，low_bits确定在cell中的位置
        if x_in_type == 'INT8':
            if y_out_type == 'INT8':
                # INT8->INT8: 8个cell，每个cell存储32个值
                lut_organized = torch.zeros((8, 32), dtype=torch.int8)
                for i in range(256):
                    high_bits = (i >> 5) & 0x7  # 高3位决定cell
                    low_bits = i & 0x1F  # 低5位决定在cell中的位置
                    lut_organized[high_bits, low_bits] = lut_values[i]
            else:  # BF16
                # INT8->BF16: 16个cell，每个cell存储16个值
                lut_organized = torch.zeros((16, 16), dtype=torch.bfloat16)
                for i in range(256):
                    high_bits = (i >> 4) & 0xF  # 高4位决定cell
                    low_bits = i & 0xF  # 低4位决定在cell中的位置
                    lut_organized[high_bits, low_bits] = lut_values[i]
        else:  # BF16输入
            if y_out_type == 'INT8':
                # BF16->INT8: 2048个cell，每个cell存储32个值
                lut_organized = torch.zeros((2048, 32), dtype=torch.int8)
                for i in range(65536):
                    high_bits = (i >> 5) & 0x7FF  # 高11位决定cell
                    low_bits = i & 0x1F  # 低5位决定在cell中的位置
                    lut_organized[high_bits, low_bits] = lut_values[i]
            else:  # BF16
                # BF16->BF16: 4096个cell，每个cell存储16个值
                lut_organized = torch.zeros((4096, 16), dtype=torch.bfloat16)
                for i in range(65536):
                    high_bits = (i >> 4) & 0xFFF  # 高12位决定cell
                    low_bits = i & 0xF  # 低4位决定在cell中的位置
                    lut_organized[high_bits, low_bits] = lut_values[i]
        
        # 返回构建好的lut
        return lut_organized
    
    @staticmethod
    def simulate_lut_lookup(x_in_tensor, lut_table, x_in_type, y_out_type):
        """模拟LUT查找过程"""
        # 将输入tensor展平
        x_flat = x_in_tensor.flatten()
        output_list = []
        
        # 重新整理LUT表的形状，按照存储方式组织
        if x_in_type == 'INT8':
            if y_out_type == 'INT8':
                # INT8->INT8: 8个cell，每个cell存储32个INT8值
                lut_reshaped = lut_table.reshape(8, 32)
            else:  # BF16
                # INT8->BF16: 16个cell，每个cell存储16个BF16值
                lut_reshaped = lut_table.reshape(16, 16)
        else:  # BF16
            if y_out_type == 'INT8':
                # BF16->INT8: 2048个cell，每个cell存储32个INT8值
                lut_reshaped = lut_table.reshape(2048, 32)
            else:  # BF16
                # BF16->BF16: 4096个cell，每个cell存储16个BF16值
                lut_reshaped = lut_table.reshape(4096, 16)
        
        for x_val in x_flat:
            if x_in_type == 'INT8':
                # INT8输入：将有符号INT8转换为无符号8位值 (0-255)
                # x_int = int(x_val.item()) if x_val.item() >= 0 else int(x_val.item()) + 256
                x_int = int(x_val.item()) & 0xFF  # 有符号转无符号：-128~127 -> 0~255
                
                if y_out_type == 'INT8':
                    # INT8 -> INT8: 高3位作偏移地址，低5位作选择信号
                    high_bits = (x_int >> 5) & 0x7  # bits [7:5] - 3位偏移地址 (0-7)
                    low_bits = x_int & 0x1F  # bits [4:0] - 5位选择信号 (0-31)
                    # 在对应的cell中选择数据
                    output_val = lut_reshaped[high_bits, low_bits]
                else:  # BF16
                    # INT8 -> BF16: 高4位作偏移地址，低4位作选择信号
                    high_bits = (x_int >> 4) & 0xF  # bits [7:4] - 4位偏移地址 (0-15)
                    low_bits = x_int & 0xF  # bits [3:0] - 4位选择信号 (0-15)
                    # 在对应的cell中选择数据
                    output_val = lut_reshaped[high_bits, low_bits]
            else:  # BF16
                # BF16输入：转换为16位整数表示
                x_int = int(x_val.view(torch.int16).item()) & 0xFFFF
                
                if y_out_type == 'INT8':
                    # BF16 -> INT8: 高11位作偏移地址，低5位作选择信号
                    high_bits = (x_int >> 5) & 0x7FF  # bits [15:5] - 11位偏移地址 (0-2047)
                    low_bits = x_int & 0x1F  # bits [4:0] - 5位选择信号 (0-31)
                    # 在对应的cell中选择数据
                    output_val = lut_reshaped[high_bits, low_bits]
                else:  # BF16
                    # BF16 -> BF16: 高12位作偏移地址，低4位作选择信号
                    high_bits = (x_int >> 4) & 0xFFF  # bits [15:4] - 12位偏移地址 (0-4095)
                    low_bits = x_int & 0xF  # bits [3:0] - 4位选择信号 (0-15)
                    # 在对应的cell中选择数据
                    output_val = lut_reshaped[high_bits, low_bits]
            
            output_list.append(output_val)
        
        # 转换回tensor并重塑为原始形状
        output_tensor = torch.stack(output_list).reshape(x_in_tensor.shape)
        
        return output_tensor
