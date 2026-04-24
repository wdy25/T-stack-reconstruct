import torch
from .prim import Primitive, PrimitiveType
from basics.addr_blocks_in_mem import AddrBlocksInMem
from basics.data_block import DataBlock
from core.simulator.emulator.core import Core as EmuCore
from core.simulator.analyser.core import Core as AnalyCore
from core.simulator.analyser.event import EventType, ComputeEvent, MemoryEvent
from core.utils.get_byte_num import get_torch_dtype_from_type_num, get_elements_num_in_cell
import math
import struct
import numpy as np

from copy import deepcopy
from myhdl import bin, intbv

class PrimAdd(Primitive):
    def __init__(self, deps, x_in_1_addr, x_in_2_addr, y_out_addr, 
                 vector_num, vector_len_in_32B, 
                 scalar, y_out_type, 
                 bc_mode, add_or_sub):
        super().__init__()
        self.name = "Add"  # 定义名称
        self.type = PrimitiveType.VECTOR  # 定义类型
        
        # 记录参数，即原语中的所有字段，包括para中某些需要额外配置的字段
        self.deps = deps
        self.x_in_1_addr = x_in_1_addr
        self.x_in_2_addr = x_in_2_addr
        self.y_out_addr = y_out_addr
        self.vector_num = vector_num
        self.vector_len_in_32B = vector_len_in_32B
        self.scalar = scalar
        self.y_out_type = y_out_type
        self.bc_mode = bc_mode
        self.add_or_sub = add_or_sub
        # torch type
        self.x_in_torch_dtype = torch.bfloat16  # 输入固定为BF16
        self.y_out_torch_dtype = get_torch_dtype_from_type_num(self.y_out_type)

        self.x_in_1_len = (self.vector_num+1) * (self.vector_len_in_32B+1)
        
        # 根据bc_mode确定第二个输入的长度
        if self.bc_mode == 0:  # 两张量逐元素相加
            self.x_in_2_len = (self.vector_num+1) * (self.vector_len_in_32B+1)
        elif self.bc_mode == 1:  # 第一个维度多播（行向量）
            self.x_in_2_len = (self.vector_len_in_32B+1)
        else:  # bc_mode == 2, 标量增减
            self.x_in_2_len = 0

        # INPUT: BF16
        # OUTPUT: INT8/BF16
        if self.y_out_type == 0:   # BF16->INT8
            self.y_out_len = (self.vector_num+1) * math.ceil((self.vector_len_in_32B+1) / 2)
        else:   # BF16->BF16
            self.y_out_len = (self.vector_num+1) * (self.vector_len_in_32B+1)
        
        # 生成PIC
        self.setPIC()
        
    def scalar_to_bf16(self, value):
        """将scalar值转换为BF16格式的16位整数"""
        # 将float转换为BF16
        # BF16格式：1位符号位 + 8位指数 + 7位尾数
        if isinstance(value, int):
            value = float(value)
        
        # 将float32转换为BF16
        # 方法：使用大端序打包，然后取前16位作为BF16
        float32_bytes = struct.pack('>f', value)  # 大端序
        bf16_int = struct.unpack('>H', float32_bytes[0:2])[0]  # 取前16位
        return bf16_int
        
    def setPIC(self):
        self.PIC = intbv(0, min=0, max=(1<<256))
        
        # 根据表格更新bit位置
        self.PIC[8:0] = 0x23        # PIC(Code): [0:7]
        self.PIC[16:8] = self.deps   # deps: [8:15]
        self.PIC[47:32] = self.x_in_1_addr  # x_in_1_addr: [32:46] - 15bit
        self.PIC[63:48] = self.x_in_2_addr  # x_in_2_addr: [48:62] - 15bit
        self.PIC[79:64] = self.y_out_addr   # y_out_addr: [64:78] - 15bit
        self.PIC[96:80] = self.vector_num   # vector_num: [80:95]
        self.PIC[112:96] = self.vector_len_in_32B  # vector_len_in_32B: [96:111]
        self.PIC[128:112] = self.scalar_to_bf16(self.scalar)  # scalar: [112:127] - 16bit BF16
        self.PIC[130:128] = self.bc_mode    # bc_mode: [128:129]
        self.PIC[130] = self.add_or_sub # add_or_sub: [130]
        self.PIC[131] = self.y_out_type # y_out_type: [131]
        
    
    def execute(self, core: EmuCore):
        # Get the input tensor
        input1_data_length = core.memory.getTensorLen((self.vector_num + 1, (self.vector_len_in_32B + 1) * 16), torch.bfloat16, (0, 1))
        
        # input1_tensor = torch.frombuffer(core.memory[self.x_in_1_addr:self.x_in_1_addr + input1_data_length].to(torch.uint8).numpy().tobytes(), dtype=torch.bfloat16).reshape((self.vector_num + 1, -1))[:, :(self.vector_len_in_32B + 1) * 32].permute(0, 1)
        input1_tensor = core.memory[self.x_in_1_addr:self.x_in_1_addr + input1_data_length].view(torch.bfloat16).reshape((self.vector_num + 1, -1))[:, :(self.vector_len_in_32B + 1) * 16].permute(0, 1)
        
        input2_data_length = 0
        input2_tensor = None
        if self.bc_mode == 0:
            input2_data_length = core.memory.getTensorLen((self.vector_num + 1, (self.vector_len_in_32B + 1) * 16), torch.bfloat16, (0, 1))
            
            # input2_tensor = torch.frombuffer(core.memory[self.x_in_2_addr:self.x_in_2_addr + input2_data_length].to(torch.uint8).numpy().tobytes(), dtype=torch.bfloat16).reshape((self.vector_num + 1, -1))[:, :(self.vector_len_in_32B + 1) * 32].permute(0, 1)
            input2_tensor = core.memory[self.x_in_2_addr:self.x_in_2_addr + input2_data_length].view(torch.bfloat16).reshape((self.vector_num + 1, -1))[:, :(self.vector_len_in_32B + 1) * 16].permute(0, 1)
            
        elif self.bc_mode == 1:
            input2_data_length = core.memory.getTensorLen(((self.vector_len_in_32B + 1) * 16,), torch.bfloat16, (0,))
            
            # input2_tensor = torch.frombuffer(core.memory[self.x_in_2_addr:self.x_in_2_addr + input2_data_length].to(torch.uint8).numpy().tobytes(), dtype=torch.bfloat16).reshape((1, -1))[:, :(self.vector_len_in_32B + 1) * 32].permute(0, 1)
            input2_tensor = core.memory[self.x_in_2_addr:self.x_in_2_addr + input2_data_length].view(torch.bfloat16).reshape((1, -1))[:, :(self.vector_len_in_32B + 1) * 16].permute(0, 1)
            
        else:  # bc_mode == 2
            scalar_bf16 = self.scalar_to_bf16(self.scalar)
            input2_tensor = torch.tensor([[scalar_bf16]], dtype=torch.bfloat16)             
        
        input1_tensor = input1_tensor.to(torch.bfloat16)
        input2_tensor = input2_tensor.to(torch.bfloat16)
        
        if self.add_or_sub == 0:
            output_tensor = input1_tensor + input2_tensor
        else:
            output_tensor = input1_tensor - input2_tensor
        
        if self.y_out_type == 0:  # BF16->INT8
            output_tensor = output_tensor.clamp(-128, 127).to(torch.int8)
        else:
            output_tensor = output_tensor.to(torch.bfloat16)
        
        # Write the output tensor to the memory
        core.memory.writeTensor(self.y_out_addr, output_tensor, (0, 1))
    
    
    def generate_events(self, core: AnalyCore):
        events = []
        real_len = np.ceil((self.vector_len_in_32B + 1) * 16 / core.config["core"]["vec_parallelism"])
        computation = (self.vector_num + 1) * real_len * core.config["core"]["vec_parallelism"]
        
        input1_hierarchy = 0 if self.x_in_1_addr < core.config["core"]["L0_memory_capacity"] // core.config["core"]["memory_width"] else 1
        input2_hierarchy = 0 if self.x_in_2_addr < core.config["core"]["L0_memory_capacity"] // core.config["core"]["memory_width"] else 1
        output_hierarchy = 0 if self.y_out_addr < core.config["core"]["L0_memory_capacity"] // core.config["core"]["memory_width"] else 1
        
        if self.bc_mode in (0, 1): # 0: two tensors element-wise add; 1: broadcast row vector add
            if self.y_out_type == 0:   # BF16->INT8
                cycles_per_cell = 7 # 2read + add + convert + write = 3 + 1 + 1 + 2 = 7 cycles per cell
            else:   # BF16->BF16
                cycles_per_cell = 6 # 2read + add + write = 3 + 1 + 2 = 6 cycles per cell
        else: # scalar add
            if self.y_out_type == 0:   # BF16->INT8
                cycles_per_cell = 6 # 1read + add + convert + write = 2 + 1 + 1 + 2 = 6 cycles per cell
            else:   # BF16->BF16
                cycles_per_cell = 5 # 1read + add + write = 2 + 1 + 2 = 5 cycles per cell
        
        total_cycle = (self.vector_num + 1) * real_len * cycles_per_cell
        
        compute_event = ComputeEvent(
            name="Add",
            parent=core.vector.full_name,
            compute_type=EventType.VECTOR,
            computation=computation,
            theoretical_computation=(self.vector_num + 1) * (self.vector_len_in_32B + 1) * 16,
            max_consume_rate=computation / total_cycle,
            energy=computation * core.config["core"]["bf16_PE_add_energy"],
        )
        events.append(compute_event)
        
        # read
        L0_volume = 0
        L1_volume = 0
        
        if self.bc_mode in (0, 1): # 0: two tensors element-wise add; 1: broadcast row vector add
            if input1_hierarchy == 0:
                L0_volume += computation * 2 # one add requires one BF16 elemet of input1 which is 2 bytes.
            else:
                L1_volume += computation * 2
            
            if input2_hierarchy == 0:
                L0_volume += computation * 2 # one add requires one BF16 elemet of input2 which is 2 bytes.
            else:
                L1_volume += computation * 2
        else: # scalar add
            if input1_hierarchy == 0:
                L0_volume += computation * 2
            else:
                L1_volume += computation * 2
            # no read input2 from memory because it's scalar.
            
        if L0_volume > 0:
            L0_read_event = MemoryEvent(
                name="Add_read_L0",
                parent=core.memory[0].full_name,
                memory_type=EventType.READ,
                volume=L0_volume,
                bounded_events=[],
                energy=np.ceil(L0_volume / core.config["core"]["memory_width"]) * core.config["core"]["L0_memory_read_energy"], 
                max_bandwidth=L0_volume / total_cycle,
                hierarchy=0
            )
            events.append(L0_read_event)
        if L1_volume > 0:
            L1_read_event = MemoryEvent(
                name="Add_read_L1",
                parent=core.memory[1].full_name,
                memory_type=EventType.READ,
                volume=L1_volume,
                bounded_events=[],
                energy=np.ceil(L1_volume / core.config["core"]["memory_width"]) * core.config["core"]["L1_memory_read_energy"], 
                max_bandwidth=L1_volume / total_cycle,
                hierarchy=1
            )
            events.append(L1_read_event)

        # write
        elements_per_vector = (self.vector_len_in_32B + 1) * get_elements_num_in_cell(self.x_in_torch_dtype)
        total_output_cells = (self.vector_num + 1) * np.ceil(elements_per_vector / get_elements_num_in_cell(self.y_out_torch_dtype))
        L0_write_volume = total_output_cells * 32 # the volume of data written to L0 memory, in bytes
        output_event = MemoryEvent(
            name="Add_write",
            parent=core.memory[output_hierarchy].full_name,
            memory_type=EventType.WRITE,
            volume=L0_write_volume,
            bounded_events=[],
            energy=(np.ceil(L0_write_volume / core.config["core"]["memory_width"]) * core.config["core"]["L0_memory_write_energy"]) if output_hierarchy == 0 else (np.ceil(L0_write_volume / core.config["core"]["L1_memory_bandwidth"]) * core.config["core"]["L1_memory_write_energy"]), 
            max_bandwidth=L0_write_volume / total_cycle,
            hierarchy=output_hierarchy
        )
        events.append(output_event)
        
        event_nms = []
        for event in events:
            event_nms.append(event.full_name)
        for event in events:
            if event.event_type in [EventType.READ, EventType.INOUT, EventType.WRITE]:
                event.bounded_events = event_nms
                event.bounded_events.remove(event.full_name)
        
        return events