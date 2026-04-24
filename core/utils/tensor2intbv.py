from typing import List, Dict, Any
import torch
from myhdl import intbv

def tensor2intbv(tensor: torch.Tensor) -> intbv:
    # Ensure the tensor is on CPU and in contiguous memory
    tensor = tensor.view(torch.uint8).view(-1).cpu().contiguous()
    
    tensor_len = tensor.numel()
    
    # Get the byte representation of the tensor
    byte_data = tensor.numpy().tobytes()
    bit_length = len(byte_data) * 8
    new_intbv = intbv(0)[bit_length:]
    
    for i in range(tensor_len):
        byte_value = tensor[i].item()
        new_intbv |= (byte_value << (i * 8))
    
    return new_intbv

def intbv2tensor(intbv_values: List[intbv], dtype=torch.uint8) -> torch.Tensor:
    all_list = []
    for intbv_value in intbv_values:
        bit_length = len(intbv_value)
        byte_length = (bit_length + 7) // 8
        assert dtype in [torch.uint8, torch.int8], "Only uint8 and int8 are now supported"
        bytes_list = []
        for i in range(byte_length):
            byte_value = int((intbv_value >> (i * 8)) & 0xFF)
            if dtype == torch.int8:
                byte_value = byte_value - 256 if byte_value >= 128 else byte_value
            bytes_list.append(byte_value)
        all_list.extend(bytes_list)
    return torch.tensor(all_list, dtype=dtype)


if __name__ == "__main__":
    # Example usage
    tensor = torch.tensor([2,3,4,5,6], dtype=torch.uint8)
    intbv_result = tensor2intbv(tensor)
    print(bin(intbv_result))