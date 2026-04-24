import torch
import numpy as np

from copy import deepcopy
from myhdl import bin, intbv

from core.ir.prims.prim import Primitive
from basics.addr_blocks_in_mem import AddrBlocksInMem
from basics.data_block import DataBlock
from core.simulator.emulator.core import Core as EmuCore
from core.simulator.analyser.core import Core as AnalyCore
from core.simulator.analyser.event import EventType, ComputeEvent, MemoryEvent
from core.utils.get_byte_num import get_torch_dtype_from_type_num, get_elements_num_in_cell, get_byte_num
import math

class PrimNonlinear(Primitive):
    def __init__(self, deps, x_in_addr, func, y_out_addr, 
                 vector_num, vector_len_32B, 
                 out_type, 
                 x_in: DataBlock=None):
        super().__init__()
        self.name = "Nonlinear"  # 定义名称
        
        # 记录参数，即原语中的所有字段，包括para中某些需要额外配置的字段
        self.deps = deps
        self.x_in_addr = x_in_addr
        self.func = func
        self.y_out_addr = y_out_addr
        self.vector_num = vector_num
        self.vector_len_32B = vector_len_32B
        self.out_type = out_type
        # torch dtype
        self.x_in_torch_dtype = torch.bfloat16  # 输入固定为BF16
        self.y_out_torch_dtype = get_torch_dtype_from_type_num(self.out_type)

        # 计算输入和输出长度
        # INPUT: BF16 (固定)
        self.x_in_len = (self.vector_num+1) * (self.vector_len_32B+1)
        
        # OUTPUT: INT8/BF16
        if self.out_type == 0:   # BF16->INT8
            self.y_out_len = (self.vector_num+1) * math.ceil((self.vector_len_32B+1) / 2)
        else:   # BF16->BF16
            self.y_out_len = (self.vector_num+1) * (self.vector_len_32B+1)

        # 保存原语需要提前配置的数据
        if x_in is not None:
            if x_in == []:
                self.data_blocks["x_in"] = DataBlock(data=None, length=self.x_in_len, zero=0, addressing="32B")
            else:
                self.data_blocks["x_in"] = deepcopy(x_in)

        # 计算地址 起始地址和数据长度
        self.setAddrBlocks()
        
        # 生成PIC
        self.setPIC()
        
    def setPIC(self):
        self.PIC = intbv(0, min=0, max=(1<<256))
        
        # 根据表格更新bit位置
        self.PIC[8:0] = 0x93        # PIC(Code): [0:7] - 非线性计算
        self.PIC[16:8] = self.deps   # deps: [8:15]
        self.PIC[47:32] = self.x_in_addr  # x_in_addr: [32:46] - 15bit
        self.PIC[64:48] = self.func       # func: [48:63] - 16bit
        self.PIC[79:64] = self.y_out_addr   # y_out_addr: [64:78] - 15bit
        self.PIC[96:80] = self.vector_num   # vector_num: [80:95] - 16bit
        self.PIC[112:96] = self.vector_len_32B  # vector_len_32B: [96:111] - position as per spec
        self.PIC[131] = self.out_type # out_type: [131]
        
    def setAddrBlocks(self):
        # 除了各种输入数据外，还需要记录结果数据的地址等各种运行中需要的数据地址和长度
        self.data_addr_list["x_in"] = AddrBlocksInMem({self.x_in_addr: self.x_in_len}, "32B")
        self.data_addr_list["y_out"] = AddrBlocksInMem({self.y_out_addr: self.y_out_len}, "32B")

    @staticmethod
    def get_func_value(func_name):
        """Get function selection value"""
        func_map = {
            "reciprocal": 0b0000_0000_0000_0001,  # 1/a
            "sqrt":       0b0000_0000_0000_0010,  # sqrt(a)
            "rsqrt":      0b0000_0000_0000_0100,  # 1/sqrt(a)
            "sin":        0b0000_0000_0000_1000,  # sin(pi*a) or sin(a)
            "cos":        0b0000_0000_0001_0000,  # cos(pi*a) or cos(a)
            "log2":       0b0000_0000_0010_0000,  # log2(a)
            "exp2":       0b0000_0000_0100_0000,  # 2^a
            "exp":        0b0000_0000_1000_0000,  # e^a
            "tanh":       0b0000_0001_0000_0000,  # tanh(a)
            "sigmoid":    0b0000_0010_0000_0000,  # sigmoid(a)
        }
        return func_map.get(func_name, 0)

    @staticmethod
    def get_supported_functions():
        """Get supported functions list"""
        # supported_functions = ["reciprocal", "sqrt", "rsqrt", "sin", "cos", "log2", "exp2", "exp", "tanh", "sigmoid"]
        supported_functions = ["sqrt", "sin", "cos", "exp"]
        return supported_functions
    
    def execute(self, core: EmuCore):
        # Input data is BF16 type, each 32B unit contains 16 BF16 elements
        # Shape: (vector_num+1, (vector_len_32B+1) * 16)
        
        # Read input tensor
        input_tensor_size = (self.vector_num + 1, (self.vector_len_32B + 1) * 16)
        input_tensor = core.memory.readTensor(
            self.x_in_addr, self.x_in_len, input_tensor_size, torch.bfloat16
        )
        
        # Select and apply nonlinear function based on func bitmask
        func = int(self.func)
        
        if func == 0b0000_0000_0000_0001:  # reciprocal: 1/a
            output = 1.0 / input_tensor
        elif func == 0b0000_0000_0000_0010:  # sqrt: sqrt(a)
            output = torch.sqrt(input_tensor)
        elif func == 0b0000_0000_0000_0100:  # rsqrt: 1/sqrt(a)
            output = torch.rsqrt(input_tensor)
        elif func == 0b0000_0000_0000_1000:  # sin: sin(pi*a)
            output = torch.sin(np.pi * input_tensor)
        elif func == 0b0000_0000_0001_0000:  # cos: cos(pi*a)
            output = torch.cos(np.pi * input_tensor)
        elif func == 0b0000_0000_0010_0000:  # log2: log2(a)
            output = torch.log2(input_tensor)
        elif func == 0b0000_0000_0100_0000:  # exp2: 2^a
            output = torch.pow(2.0, input_tensor)
        elif func == 0b0000_0000_1000_0000:  # exp: e^a
            output = torch.exp(input_tensor)
        elif func == 0b0000_0001_0000_0000:  # tanh: tanh(a)
            output = torch.tanh(input_tensor)
        elif func == 0b0000_0010_0000_0000:  # sigmoid: sigmoid(a)
            output = torch.sigmoid(input_tensor)
        else:
            # Unknown function code, raise exception
            raise ValueError(f"Unsupported function code: {func:#06x}")
        
        # Convert output type
        if self.out_type == 0:  # INT8
            output = output.clamp(-128, 127).to(torch.int8)
        else:  # BF16
            output = output.to(torch.bfloat16)
        
        # Write the output tensor to memory
        core.memory.writeTensor(self.y_out_addr, output, (0, 1))
    
    def generate_events(self, core: AnalyCore):
        events = []
        elements_per_vector = (self.vector_len_32B + 1) * get_elements_num_in_cell(self.x_in_torch_dtype)
        total_elements = (self.vector_num + 1) * elements_per_vector
        computation = total_elements
        # only BF16 -> BF16
        cycles_per_cell = 4 # read + nonlinear + write = 2 + 1 + 1 = 4 cycles per cell
        total_cycle = (self.vector_num + 1) * (self.vector_len_32B + 1) * cycles_per_cell
        
        compute_event = ComputeEvent(
            name="Nonlinear",
            parent=core.vector.full_name,
            compute_type=EventType.VECTOR,
            computation=computation,
            theoretical_computation=(self.vector_num + 1) * elements_per_vector,
            max_consume_rate=computation / total_cycle,
            energy=computation * core.config["core"]["bf16_PE_energy"], # use BF16 PE energy as BF16 nonlinear energy
        )
        events.append(compute_event)
        
        # read
        L0_volume = total_elements * get_byte_num(self.x_in_torch_dtype) # the volume of data read from L0, in bytes
            
        L0_read_event = MemoryEvent(
            name="Nonlinear_read_L0",
            parent=core.memory[0].full_name,
            memory_type=EventType.READ,
            volume=L0_volume,
            bounded_events=[],
            energy=np.ceil(L0_volume / core.config["core"]["memory_width"]) * core.config["core"]["L0_memory_read_energy"], 
            max_bandwidth=L0_volume / total_cycle,
            hierarchy=0
        )
        events.append(L0_read_event)

        # write
        total_output_cells = (self.vector_num + 1) * np.ceil(elements_per_vector / get_elements_num_in_cell(self.y_out_torch_dtype))
        L0_write_volume = total_output_cells * 32 # the volume of data written to L0 memory, in bytes
        
        L0_write_event = MemoryEvent(
            name="Nonlinear_write_L0",
            parent=core.memory[0].full_name,
            memory_type=EventType.WRITE,
            volume=L0_write_volume,
            bounded_events=[],
            energy=(np.ceil(L0_write_volume / core.config["core"]["memory_width"]) * core.config["core"]["L0_memory_write_energy"]), 
            max_bandwidth=L0_write_volume / total_cycle,
            hierarchy=0
        )
        events.append(L0_write_event)
        
        event_nms = []
        for event in events:
            event_nms.append(event.full_name)
        for event in events:
            if event.event_type in [EventType.READ, EventType.INOUT, EventType.WRITE]:
                event.bounded_events = event_nms
                event.bounded_events.remove(event.full_name)
        
        return events
