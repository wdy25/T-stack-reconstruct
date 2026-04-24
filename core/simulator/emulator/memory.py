import torch
from functools import lru_cache
import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Tuple


class Memory:
    def __init__(self, capacity_in_byte: int, width_in_byte: int):
        self.capacity_in_byte = int(capacity_in_byte)
        self.width_in_byte = int(width_in_byte)
        
        assert capacity_in_byte % width_in_byte == 0, "The capacity should be a multiple of the width"
        
        self.memory = torch.zeros((int(capacity_in_byte // width_in_byte), int(width_in_byte)), dtype=torch.uint8)
        
        
    def __getitem__(self, index):
        if isinstance(index, slice):
            assert 0 <= index.start < len(self), "The start index is out of range"
            assert 0 <= index.stop < len(self), "The stop index is out of range"
            start = int(index.start) if index.start is not None else None
            stop = int(index.stop) if index.stop is not None else None
            step = int(index.step) if index.step is not None else None
            return self.memory[start:stop:step, :]
        else:
            assert 0 <= index < len(self), "The index is out of range"
            return self.memory[index, :]    
    
    
    def getTensorLen(self, tensor_size, data_type: torch.dtype, order: tuple) -> int:
        """Compute how many memory cells (rows) are needed to store a tensor.

        A "cell" here means one memory row of width ``self.width_in_byte``.
        For a given ``data_type``, one cell can hold ``getNumberInCell(data_type)``
        elements (for ``torch.bool``, this is counted in bits).

        Args:
            tensor_size: Original tensor shape (e.g. ``tensor.shape``).
            data_type: Tensor dtype (e.g. ``torch.bfloat16``).
            order: Permutation used for storage.

        Returns:
            Number of memory cells (rows) required to store the tensor.
        """
        assert(len(tensor_size) == len(order))
        
        number_in_cell = self.getNumberInCell(data_type)

        permuted_size = [tensor_size[i] for i in order]
        inner_dimension = permuted_size[-1]
        dimension_padded_to_cell = np.ceil(inner_dimension / number_in_cell) * number_in_cell
        
        if len(permuted_size) == 1:
            tensor_len = dimension_padded_to_cell // number_in_cell
        else:
            tensor_len = np.prod(permuted_size[:-1]) * dimension_padded_to_cell // number_in_cell
        
        return int(tensor_len)
    
    
    def write8B(self, idx_8B: int, data: torch.Tensor) -> None:
        cell_idx = idx_8B * 8 // self.width_in_byte
        byte_offset = (idx_8B * 8) % self.width_in_byte
        assert byte_offset + 8 <= self.width_in_byte, "Data exceeds memory cell width"
        self.memory[cell_idx][byte_offset:byte_offset+8] = data.view(torch.uint8)
    
    
    def write1B(self, idx_1B: int, data: torch.Tensor) -> None:
        cell_idx = idx_1B // self.width_in_byte
        byte_offset = idx_1B % self.width_in_byte
        self.memory[cell_idx][byte_offset] = data
    
    
    def read1B(self, idx_1B: int) -> torch.Tensor:
        cell_idx = idx_1B // self.width_in_byte
        byte_offset = idx_1B % self.width_in_byte
        return self.memory[cell_idx][byte_offset]
        
    
    def writeTensor(self, index: int, tensor: torch.Tensor, order: Optional[tuple] = None) -> None:
        if order is None:
            order = tuple(range(len(tensor.shape)))
        assert(len(tensor.shape) == len(order))

        data_type = tensor.dtype
        
        # tensor = tensor.to(dtype=data_type)
        permuted_x = tensor.permute(order)
        
        number_in_cell = self.getNumberInCell(data_type)


        # 计算内部维度的填充到32位单元的数量
        inner_dimension = permuted_x.shape[-1]
        # 在Python中，负数索引表示从序列的末尾开始计数。因此，-1 表示最后一个元素，-2 表示倒数第二个元素，以此类推
        dimension_padded_to_cell = np.ceil(inner_dimension / number_in_cell) * number_in_cell

        zero_to_full_cell = torch.zeros(size=[dim for dim in permuted_x.shape[:-1]] + [int(dimension_padded_to_cell) - inner_dimension]).to(data_type)
        # 补0操作   # 使用[:-1]获取除最后一个元素外的 子序列

        # 将零张量连接到 permuted_x，以确保其维度被填充到32位单元
        permuted_x = torch.cat((permuted_x, zero_to_full_cell), dim=-1)
        
        # Reshape the tensor to the shape of the memory cell
        permuted_x = permuted_x.reshape(-1, number_in_cell)
        assert permuted_x.shape[0] == self.getTensorLen(tensor.shape, data_type, order)
        
        # Write the tensor to the memory
        if data_type == torch.bool:
            permuted_x = permuted_x.reshape(permuted_x.shape[0], -1, 8).to(torch.uint8)
            assert permuted_x.shape[1] == self.width_in_byte
            for i in range(permuted_x.shape[0]):
                for j in range(permuted_x.shape[1]):
                    self.memory[index + i, j] = int(''.join([str(int(x)) for x in permuted_x[i, j, :].tolist()[::-1]]), 2)
        else:
            for i in range(permuted_x.shape[0]):
                # self.memory[index + i, :] = torch.Tensor(permuted_x[i, :].clone().untyped_storage().tolist()).to(torch.uint8)
                self.memory[index + i, :] = permuted_x[i, :].view(torch.uint8)
    
    
    def readTensor(self, addr: int, length:int, tensor_size, data_type: torch.dtype, order: Optional[tuple] = None) -> torch.Tensor:
        if order is None:
            order = tuple(range(len(tensor_size)))
        assert(len(tensor_size) == len(order))
        # reverse the process above
        number_in_cell = self.getNumberInCell(data_type)
        permuted_size = [tensor_size[i] for i in order]

        # 计算内部维度的填充，与 writeTensor 逻辑一致
        inner_dimension = permuted_size[-1]
        dimension_padded_to_cell = int(np.ceil(inner_dimension / number_in_cell) * number_in_cell)

        # 计算在内存中占据的行数 (tensor_len)
        if len(permuted_size) == 1:
            tensor_len = dimension_padded_to_cell // number_in_cell
        else:
            tensor_len = np.prod(permuted_size[:-1]) * dimension_padded_to_cell // number_in_cell
            
        assert length == tensor_len, "The length does not match the tensor size and data type"

        # 从内存中读取原始字节
        read_data = self.memory[addr : addr + int(tensor_len), :].contiguous()

        if data_type == torch.bool:
            # writeTensor 中使用了 [::-1] 反转位序并转 int，实际上对应 Little Endian (Index 0 是 LSB)
            # 使用 numpy 的 unpackbits(..., bitorder='little') 进行还原
            np_data = read_data.cpu().numpy()
            unpacked = np.unpackbits(np_data, axis=1, bitorder='little')
            tensor_data = torch.from_numpy(unpacked.copy()).to(torch.bool)
        else:
            # 其他类型直接使用 view 转换视图
            tensor_data = read_data.view(data_type)

        # 重塑为填充后的形状
        expected_shape = list(permuted_size[:-1]) + [dimension_padded_to_cell]
        tensor_data = tensor_data.reshape(expected_shape)

        # 消除末尾的填充
        if dimension_padded_to_cell != inner_dimension:
            tensor_data = tensor_data[..., :inner_dimension]

        # 反转维度排列
        reverse_order = [order.index(i) for i in range(len(order))]
        tensor_data = tensor_data.permute(reverse_order)
        
        return tensor_data
            

    def __len__(self):
        return self.capacity_in_byte // self.width_in_byte
    
    @lru_cache(maxsize=None)
    def getNumberInCell(self, data_type: torch.dtype) -> int:
        if (data_type == torch.float16 or data_type == torch.bfloat16):
            return self.width_in_byte // 2
        elif (data_type == torch.float32 or data_type == torch.int32):
            return self.width_in_byte // 4
        elif (data_type == torch.int8 or data_type == torch.uint8):
            return self.width_in_byte
        elif (data_type == torch.bool):
            return self.width_in_byte * 8
    
    def convertTensorToBytes(self, tensor: torch.Tensor, order: tuple) -> torch.Tensor:
        assert(len(tensor.shape) == len(order))

        data_type = tensor.dtype
        
        # tensor = tensor.to(dtype=data_type)
        permuted_x = tensor.permute(order)
        
        number_in_cell = self.getNumberInCell(data_type)


        # 计算内部维度的填充到32位单元的数量
        inner_dimension = permuted_x.shape[-1]
        # 在Python中，负数索引表示从序列的末尾开始计数。因此，-1 表示最后一个元素，-2 表示倒数第二个元素，以此类推
        dimension_padded_to_cell = np.ceil(inner_dimension / number_in_cell) * number_in_cell

        zero_to_full_cell = torch.zeros(size=[dim for dim in permuted_x.shape[:-1]] + [int(dimension_padded_to_cell) - inner_dimension]).to(data_type)
        # 补0操作   # 使用[:-1]获取除最后一个元素外的 子序列

        # 将零张量连接到 permuted_x，以确保其维度被填充到32位单元
        permuted_x = torch.cat((permuted_x, zero_to_full_cell), dim=-1)
        
        # Reshape the tensor to the shape of the memory cell
        permuted_x = permuted_x.reshape(-1, number_in_cell)
        assert permuted_x.shape[0] == self.getTensorLen(tensor.shape, data_type, order)
        
        # Write the tensor to the memory
        tmp = torch.zeros(permuted_x.shape[0], self.width_in_byte).to(torch.uint8)
        if data_type == torch.bool:
            permuted_x = permuted_x.reshape(permuted_x.shape[0], -1, 8).to(torch.uint8)
            assert permuted_x.shape[1] == self.width_in_byte
            for i in range(permuted_x.shape[0]):
                for j in range(permuted_x.shape[1]):
                    tmp[i, j] = int(''.join([str(int(x)) for x in permuted_x[i, j, :].tolist()[::-1]]), 2)
        else:
            for i in range(permuted_x.shape[0]):
                tmp[i, :] = torch.Tensor(permuted_x[i, :].clone().untyped_storage().tolist()).to(torch.uint8)
        
        return tmp
    

if __name__ == "__main__":
    mem = Memory(1024 * 1024, 32)
    # print(mem.getTensorLen((2, 2, 65), torch.bfloat16, (0, 1, 2)))
    a = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], dtype=torch.bfloat16)
    mem.writeTensor(0, a, order=(0,))
    a_tensor = mem.readTensor(0, mem.getTensorLen((17,), torch.bfloat16, (0,)), (17,), torch.bfloat16)
    print(torch.sum(abs(a_tensor - a)))
    # print(mem.convertTensorToBytes(a, (0,)))
    # b = torch.tensor([1,1,1,0,0,0,1,1,0,1,1,1,1,1,0,1], dtype=torch.bool)
    # print(mem.convertTensorToBytes(b, (0,)))
    # print(b[1])
    # 模拟dendrite deep convolution在memory中的存取
    bs, c_in, x_in_h, x_in_w = 1, 16, 8, 8
    x_shape = (bs,c_in,x_in_h,x_in_w)
    x_in = torch.rand(x_shape, dtype=torch.bfloat16)
    # print(x_in)
    # NHWC 格式存储
    mem.writeTensor(0, x_in, order=(0,2,3,1))
    x_in_tensor = mem.readTensor(0, mem.getTensorLen(x_shape, torch.bfloat16, (0,2,3,1)), x_shape, torch.bfloat16, order=(0,2,3,1))
    print(torch.sum(abs(x_in_tensor - x_in)))