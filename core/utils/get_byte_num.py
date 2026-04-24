import torch
from functools import lru_cache
from typing import Union

from core.ir.data import Data, DataType

@lru_cache(maxsize=None)
def get_byte_num(data_type: torch.dtype) -> Union[int, float]:
    if (data_type == torch.float16 or data_type == torch.bfloat16):
        return 2
    elif (data_type == torch.float32 or data_type == torch.int32):
        return 4
    elif (data_type == torch.int8 or data_type == torch.uint8):
        return 1
    elif (data_type == torch.bool):
        return 0.125
    
@lru_cache(maxsize=None)
def get_elements_num_in_cell(data_type: torch.dtype) -> int:
    if (data_type == torch.float16 or data_type == torch.bfloat16):
        return 16
    elif (data_type == torch.float32 or data_type == torch.int32):
        return 8
    elif (data_type == torch.int8 or data_type == torch.uint8):
        return 32
    elif (data_type == torch.bool):
        return 256

@lru_cache(maxsize=None)
def get_torch_dtype_from_DataType(in_dtype: DataType) -> torch.dtype:
    dtype_map = {
                    DataType.INT8: torch.int8,
                    DataType.BF16: torch.bfloat16,
                    DataType.SPIKE: torch.bool,  # Assuming spike data is stored as uint8
                    DataType.INT32: torch.int32,
                }
    return dtype_map.get(in_dtype)

# only for soma
@lru_cache(maxsize=None)
def get_torch_dtype_from_type_num(type_num: int) -> torch.dtype:
    if type_num == 0:
        return torch.int8
    elif type_num == 1:
        return torch.bfloat16
    elif type_num == 2:
        return torch.bool