import torch
from core.ir.prims.prim import Primitive
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

class PrimMultiply(Primitive):
    def __init__(self, deps, x_in_1_addr, x_in_2_addr, y_out_addr, 
                 vector_num, vector_len_in_32B, 
                 scalar, y_out_type, 
                 bc_mode, mult_or_div, 
                 x_in_1: DataBlock=None, x_in_2: DataBlock=None):
        super().__init__()
        self.name = "Multiply"  # 定义名称
        
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
        self.mult_or_div = mult_or_div
        # torch type
        self.x_in_torch_dtype = torch.bfloat16  # 输入固定为BF16
        self.y_out_torch_dtype = get_torch_dtype_from_type_num(self.y_out_type)

        self.x_in_1_len = (self.vector_num+1) * (self.vector_len_in_32B+1)
        
        # 根据bc_mode确定第二个输入的长度
        if self.bc_mode == 0:  # 两张量逐元素相乘
            self.x_in_2_len = (self.vector_num+1) * (self.vector_len_in_32B+1)
        elif self.bc_mode == 1:  # 第一个维度多播（行向量）
            self.x_in_2_len = (self.vector_len_in_32B+1)
        else:  # bc_mode == 2, 标量缩放
            self.x_in_2_len = 1

        # INPUT: BF16
        # OUTPUT: INT8/BF16
        if self.y_out_torch_dtype == torch.int8:   # BF16->INT8
            self.y_out_len = (self.vector_num+1) * math.ceil((self.vector_len_in_32B+1) / 2)
        else:   # BF16->BF16
            self.y_out_len = (self.vector_num+1) * (self.vector_len_in_32B+1)

        # 保存原语需要提前配置的数据，如果有para则必须在此生成
        # 对于非para的数据，可以有3种情况：
        # 1. 为None，表示不需要提前配置，可能是其他原语的输出作为输入，直接跳过这份数据即可
        # 2. 为[]，表示需要提前配置，需要生成一个DataBlock，长度根据原语参数确定，zero=0表示全随机
        # 3. 为DataBlock，表示直接使用这个DataBlock 
        
        if x_in_1 is not None:
            if x_in_1 == []:
                self.data_blocks["x_in_1"] = DataBlock(data=None, length=1, zero=0, addressing="32B")
            else:
                self.data_blocks["x_in_1"] = deepcopy(x_in_1)
        
        if x_in_2 is not None:
            if x_in_2 == []:
                self.data_blocks["x_in_2"] = DataBlock(data=None, length=1, zero=0, addressing="32B")
            else:
                self.data_blocks["x_in_2"] = deepcopy(x_in_2)

        # 计算地址 起始地址和数据长度
        self.setAddrBlocks()
        
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
        self.PIC[8:0] = 0x13        # PIC(Code): [0:7]
        self.PIC[16:8] = self.deps   # deps: [8:15]
        self.PIC[47:32] = self.x_in_1_addr  # x_in_1_addr: [32:46] - 15bit
        self.PIC[63:48] = self.x_in_2_addr  # x_in_2_addr: [48:62] - 15bit
        self.PIC[79:64] = self.y_out_addr   # y_out_addr: [64:78] - 15bit
        self.PIC[96:80] = self.vector_num   # vector_num: [80:95]
        self.PIC[112:96] = self.vector_len_in_32B  # vector_len_in_32B: [96:111]
        self.PIC[128:112] = self.scalar_to_bf16(self.scalar)  # scalar: [112:127] - 16bit BF16
        self.PIC[130:128] = self.bc_mode    # bc_mode: [128:129]
        self.PIC[130] = self.mult_or_div # mult_or_div: [130]
        self.PIC[131] = self.y_out_type # y_out_type: [131]
        
    def setAddrBlocks(self):
        # 除了各种输入数据外，还需要记录结果数据的地址、Para的地址等各种运行中需要的数据地址核长度
        # 此函数不应该直接使用self.data_blocks中的数据长度，因为这里计算的长度需要和self.data_blocks中的数据长度进行对比以确保传入的数据长度是正确的

        self.data_addr_list["x_in_1"] = AddrBlocksInMem({self.x_in_1_addr: self.x_in_1_len}, "32B")
        self.data_addr_list["x_in_2"] = AddrBlocksInMem({self.x_in_2_addr: self.x_in_2_len}, "32B")
        self.data_addr_list["y_out"] = AddrBlocksInMem({self.y_out_addr: self.y_out_len}, "32B")
    
    def execute(self, core: EmuCore):
        # Input data is BF16 type, each 32B unit contains 16 BF16 elements
        # Shape: (vector_num+1, (vector_len_in_32B+1) * 16)
        
        # Read the first input tensor
        input1_tensor_size = (self.vector_num + 1, (self.vector_len_in_32B + 1) * 16)
        input1_tensor = core.memory.readTensor(
            self.x_in_1_addr, self.x_in_1_len, input1_tensor_size, torch.bfloat16
        )
        
        # Read the second input or use a scalar based on bc_mode
        if self.bc_mode == 0:  # Element-wise operation for two tensors
            input2_tensor_size = (self.vector_num + 1, (self.vector_len_in_32B + 1) * 16)
            input2_tensor = core.memory.readTensor(
                self.x_in_2_addr, self.x_in_2_len, input2_tensor_size, torch.bfloat16
            )
        elif self.bc_mode == 1:  # Broadcast on the first dimension (row vector)
            input2_tensor_size = ((self.vector_len_in_32B + 1) * 16,)
            input2_tensor = core.memory.readTensor(
                self.x_in_2_addr, self.x_in_2_len, input2_tensor_size, torch.bfloat16
            )
            # Unsqueeze to broadcastable shape (1, channels)
            input2_tensor = input2_tensor.unsqueeze(0)
        else:  # bc_mode == 2, scalar operation
            input2_tensor = torch.tensor([[self.scalar]], dtype=torch.bfloat16)
        
        # Execute multiplication or division
        if self.mult_or_div == 0:  # Multiplication
            output = input1_tensor * input2_tensor
        else:  # Division
            output = input1_tensor / input2_tensor
        
        # Convert output type
        if self.y_out_torch_dtype == torch.int8:  # INT8
            output = output.clamp(-128, 127).to(torch.int8)
        else:  # BF16
            output = output.to(torch.bfloat16)
        
        # Write the output tensor to the memory
        core.memory.writeTensor(self.y_out_addr, output, (0, 1))
    
    def generate_events(self, core: AnalyCore):
        events = []
        real_len = np.ceil((self.vector_len_in_32B + 1) * 16 / core.config["core"]["vec_parallelism"])
        computation = (self.vector_num + 1) * real_len * core.config["core"]["vec_parallelism"]
        
        input1_hierarchy = 0 if self.x_in_1_addr < core.config["core"]["L0_memory_capacity"] // core.config["core"]["memory_width"] else 1
        input2_hierarchy = 0 if self.x_in_2_addr < core.config["core"]["L0_memory_capacity"] // core.config["core"]["memory_width"] else 1
        output_hierarchy = 0 if self.y_out_addr < core.config["core"]["L0_memory_capacity"] // core.config["core"]["memory_width"] else 1
        
        if self.bc_mode in (0, 1): # 0: two tensors element-wise multiply; 1: row vector broadcast multiply
            if self.y_out_torch_dtype == torch.int8:   # BF16->INT8
                cycles_per_cell = 6 # 2read + multiply + convert + write = 3 + 1 + 1 + 1 = 6 cycles per cell
            else:   # BF16->BF16
                cycles_per_cell = 5 # 2read + multiply + write = 3 + 1 + 1 = 5 cycles per cell
        else: # scalar multiply
            if self.y_out_torch_dtype == torch.int8:   # BF16->INT8
                cycles_per_cell = 5 # 1read + multiply + convert + write = 2 + 1 + 1 + 1 = 5 cycles per cell
            else:   # BF16->BF16
                cycles_per_cell = 4 # 1read + multiply + write = 2 + 1 + 1 = 4 cycles per cell
        
        total_cycle = (self.vector_num + 1) * real_len * cycles_per_cell
        
        compute_event = ComputeEvent(
            name="Multiply",
            parent=core.vector.full_name,
            compute_type=EventType.VECTOR,
            computation=computation,
            theoretical_computation=(self.vector_num + 1) * (self.vector_len_in_32B + 1) * 16,
            max_consume_rate=computation / total_cycle,
            energy=computation * core.config["core"]["bf16_PE_multiply_energy"],
        )
        events.append(compute_event)
        
        # read
        L0_volume = 0
        L1_volume = 0
        
        if self.bc_mode in (0, 1): # 0: two tensors element-wise multiply; 1: broadcast row vector multiply
            if input1_hierarchy == 0:
                L0_volume += computation * 2 # one multiply requires one BF16 elemet of input1 which is 2 bytes.
            else:
                L1_volume += computation * 2
            
            if input2_hierarchy == 0:
                L0_volume += computation * 2 # one multiply requires one BF16 elemet of input2 which is 2 bytes.
            else:
                L1_volume += computation * 2
        else: # scalar multiply
            if input1_hierarchy == 0:
                L0_volume += computation * 2
            else:
                L1_volume += computation * 2
            # no read input2 from memory because it's scalar.
            
        if L0_volume > 0:
            L0_read_event = MemoryEvent(
                name="Multiply_read_L0",
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
                name="Multiply_read_L1",
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
            name="Multiply_write",
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
