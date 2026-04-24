from basics.utils import *

import torch
import numpy as np
import struct
from myhdl import bin, intbv
from collections.abc import Iterable
import random
from functools import lru_cache
from copy import deepcopy


def toHexStr(x, bits=256) -> str:
    assert isinstance(x, int) or isinstance(x, intbv)
    return hex(x)[2:].zfill(bits//4)


def toBinStr(x, bits=256) -> str:
    assert isinstance(x, int) or isinstance(x, intbv)
    return bin(x).zfill(bits)


def randIntbv(bits=256) -> intbv:
    return intbv(random.getrandbits(bits), min=0, max=(1<<bits))


class DataBlock():
    def __init__(self, data=None, length: int=0, zero: int=1, addressing="32B"):
        self.addressing = addressing
        if (data is not None):
            self.data = []
            if isinstance(data, intbv):
                self.data_check(data)
                self.data.append(data)
            elif isinstance(data, Iterable):
                for d in data:
                    self.data_check(d)
                    self.data.append(d)
            elif isinstance(data, DataBlock):
                self.data = deepcopy(data.data)
        else:
            if (zero):
                self.data = [intbv(0, min=0, max=(1<<self.bits)) for _ in range(length)]
            else:
                self.data = [randIntbv(self.bits) for _ in range(length)]
        
    def __len__(self):
        return len(self.data)
    
    @property
    @lru_cache
    def bits(self):
        if self.addressing == "32B":
            return 256
        elif self.addressing == "8B":
            return 64
        elif self.addressing == "1B":
            return 8
        else:
            raise ValueError("Invalid addressing")
    
    def data_check(self, data):
        assert isinstance(data, intbv)
        assert 0 <= data < (1<<self.bits)

    def __getitem__(self, key: int):
        return self.data[key]
    
    def __setitem__(self, key: int, value: intbv):
        self.data_check(value)
        self.data[key] = value
        
    def __str__(self):
        return '\n'.join(toHexStr(x, self.bits) for x in self.data)
    
    def to_str_list(self, radix=16):
        if radix == 16:
            return [toHexStr(x, self.bits) for x in self.data]
        elif radix == 2:
            return [toBinStr(x, self.bits) for x in self.data]
        else:
            raise ValueError("Invalid radix")


def readDataBlockFromFile(filename: str, addressing: str="32B") -> DataBlock:
    if addressing != "32B":
        raise NotImplementedError("Not implemented yet")
    with open(filename, 'r') as f:
        data = []
        for line in f:
            line = line.strip()
            assert len(line) == 64
            intbv_line = intbv(int(line, 16), min=0, max=(1<<256))
            data.append(intbv_line)
        return DataBlock(data=data, addressing=addressing)


def convertTensorToDataBlock(x: torch.Tensor, type: str, order: tuple, addressing: str="32B") -> DataBlock:
    if addressing != "32B":
        raise NotImplementedError("Not implemented yet")
    
    assert(len(x.shape) == len(order))
    #print('len(x.shape)', x.shape)
    #print('len(order)', len(order))
    assert(type == 'BF16' or type == 'FP32' or type == 'INT8' or type == 'INT32' or type == 'BIN')
    #将输入张量转换为指定的数据类型
    if (type == 'BF16'):
        x = x.to(torch.bfloat16)
    elif (type == 'FP32'):
        x = x.to(torch.float32)
    elif (type == 'INT8'):
        x = x.to(torch.int8)
    elif (type == 'INT32'):
        x = x.to(torch.int32)
    elif (type == 'BIN'):
        x = x.to(torch.bool)
    
    # Permute the dimensions of x based on the order specified in the tuple
    # 根据元组中指定的顺序重新排列张量的维度
    permuted_x = x.permute(order)
    
    if (type == 'BF16'):
        number_in_32B = 16.0
    elif (type == 'FP32'):
        number_in_32B = 8.0
    elif (type == 'INT8'):
        number_in_32B = 32.0
    elif (type == 'INT32'):
        number_in_32B = 8.0
    elif (type == 'BIN'):
        number_in_32B = 256.0

    # 计算内部维度的填充到32位单元的数量
    inner_dimension = permuted_x.shape[-1]
    # 在Python中，负数索引表示从序列的末尾开始计数。因此，-1 表示最后一个元素，-2 表示倒数第二个元素，以此类推
    dimension_padded_to_cell = np.ceil(inner_dimension / number_in_32B) * number_in_32B

    zero_to_full_cell = torch.zeros(size=[dim for dim in permuted_x.shape[:-1]] + [int(dimension_padded_to_cell) - inner_dimension])
    # 补0操作   # 使用[:-1]获取除最后一个元素外的 子序列

    # 将零张量连接到 permuted_x，以确保其维度被填充到32位单元
    permuted_x = torch.cat((permuted_x, zero_to_full_cell), dim=-1)
    
    # Flatten the tensor and get the sorted indices
    flat_x = torch.flatten(permuted_x)  # 使用torch.flatten函数将张量permuted_x展平为一维张量flat_x。这意味着所有维度上的元素都被拉平，得到一个包含所有元素的一维张量
    if (not type == 'BF16' and not type == 'FP32'):
        flat_x = flat_x.to(torch.int32)
    length = flat_x.shape[0]   # 因为拉平所以用 shape[0] 表示长度个数

    x_in_mem = torch.reshape(flat_x, (int(length / number_in_32B), int(number_in_32B)))
    
    data = []
    for cell in x_in_mem:
        cell_str = ""
        if (type == 'BIN'):
            for ii in range(int(len(cell)/4)):
                cell_str = num_to_hex(cell[ii*4:ii*4+4], type) + cell_str
        else:
            for element in cell:
                cell_str = num_to_hex(element, type) + cell_str
        intbv_line = intbv(int(cell_str, 16), min=0, max=(1<<256))
        data.append(intbv_line)
    return DataBlock(data=data, addressing=addressing)


def convertWeightToDataBlock(x: torch.Tensor, type: str, addressing: str="32B") -> DataBlock:
    return convertTensorToDataBlock(x, type, (2, 3, 1, 0), addressing)


def convertFeatureToDataBlock(x: torch.Tensor, type: str, addressing: str="32B") -> DataBlock:
    return convertTensorToDataBlock(x, type, (0, 2, 3, 1), addressing)


def calculateTensorAddressLength(shape, data_type: str, addressing: str = "32B") -> int:
    """
    计算张量在指定寻址方式下占据的地址空间长度
    
    Args:
        shape: 张量维度，tuple或list类型，如(2, 3, 4, 5)
        data_type: 数据类型，支持'BF16', 'FP32', 'INT8', 'INT32', 'BIN'
        addressing: 寻址方式，目前只支持"32B"
    
    Returns:
        int: 地址空间长度（以addressing单位计算）
    """
    if addressing != "32B":
        raise NotImplementedError("Currently only 32B addressing is supported")
    
    # 验证数据类型
    assert data_type in ['BF16', 'FP32', 'INT8', 'INT32', 'BIN'], f"Unsupported data type: {data_type}"
    
    # 确定每个32B单元能存储多少个元素
    if data_type == 'BF16':
        number_in_32B = 16.0  # 32B / 2B = 16个BF16元素
    elif data_type == 'FP32':
        number_in_32B = 8.0   # 32B / 4B = 8个FP32元素
    elif data_type == 'INT8':
        number_in_32B = 32.0  # 32B / 1B = 32个INT8元素
    elif data_type == 'INT32':
        number_in_32B = 8.0   # 32B / 4B = 8个INT32元素
    elif data_type == 'BIN':
        number_in_32B = 256.0 # 32B = 256个bit
    
    # 计算张量总元素数量
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    
    # 计算最后一个维度填充后的大小
    inner_dimension = shape[-1]
    dimension_padded_to_cell = np.ceil(inner_dimension / number_in_32B) * number_in_32B
    
    # 计算填充后的总元素数量
    padded_total_elements = total_elements // inner_dimension * int(dimension_padded_to_cell)
    
    # 计算需要的32B地址单元数量
    address_length = int(np.ceil(padded_total_elements / number_in_32B))

    assert address_length == int(np.ceil(inner_dimension / number_in_32B) * np.prod(shape[:-1])), \
        f"Address length mismatch: {address_length} != {np.ceil(inner_dimension / number_in_32B) * np.prod(shape[:-1])}"
    
    return address_length


if __name__ == "__main__":
    a = intbv(0, min=0, max=(1<<256))
    print(toBinStr(a))
    # print(type(a))
    # print(bin(a.max))
    # print(str(a))
    # print(toHexStr(a.max))
    # print(toHexStr(random.getrandbits(256)))
    bb = DataBlock(length=10, zero=0, addressing="32B")
    cc = DataBlock(length=10, zero=1, addressing="32B")
    print(bb)
    print(cc)
    
    dd = DataBlock(length=1, zero=0, addressing="32B")
    print(dd)
    
    print(readDataBlockFromFile("test.txt"))
    
    
