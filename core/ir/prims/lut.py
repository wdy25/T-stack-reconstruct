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

class PrimMeanLUT(Primitive):
    def __init__(self, deps, x_in_addr, lut_in_addr, y_out_addr, 
                 vector_num, vector_len_in_32B, x_in_type, y_out_type, 
                 x_in: DataBlock=None, lut_in: DataBlock=None):
        super().__init__()
        self.name = "LUT"  # 定义名称
        
        # 记录参数，即原语中的所有字段，包括para中某些需要额外配置的字段
        self.deps = deps
        self.x_in_addr = x_in_addr
        self.lut_in_addr = lut_in_addr
        self.y_out_addr = y_out_addr
        self.vector_num = vector_num
        self.vector_len_in_32B = vector_len_in_32B
        self.x_in_type = x_in_type
        self.y_out_type = y_out_type
        # torch type
        self.x_in_torch_dtype = get_torch_dtype_from_type_num(self.x_in_type)
        self.y_out_torch_dtype = get_torch_dtype_from_type_num(self.y_out_type)

        # 计算输入数据长度
        self.x_in_len = (self.vector_num+1) * (self.vector_len_in_32B+1)

        # 计算输出数据长度，根据输入输出类型确定
        if self.x_in_type == self.y_out_type:  # 相同类型
            self.y_out_len = (self.vector_num+1) * (self.vector_len_in_32B+1)
        elif self.x_in_type == 0 and self.y_out_type == 1:  # INT8 -> BF16
            self.y_out_len = (self.vector_num+1) * (self.vector_len_in_32B+1) * 2
        elif self.x_in_type == 1 and self.y_out_type == 0:  # BF16 -> INT8
            self.y_out_len = (self.vector_num+1) * math.ceil((self.vector_len_in_32B+1) / 2)

        # LUT表的长度计算，需要根据输入数据位宽和输出数据类型确定
        # LUT表用于将输入值映射到输出值，输入为INT8或BF16，输出为INT8或BF16
        if self.x_in_type == 0:  # 输入是INT8 (8位)，LUT表需要2^8=256个条目
            if self.y_out_type == 0:  # 输出INT8
                self.lut_in_len = 256 // 32  # 256个INT8值，每32B存储32个INT8
            else:  # 输出BF16
                self.lut_in_len = 256 // 16  # 256个BF16值，每32B存储16个BF16
        else:  # 输入是BF16 (16位)，LUT表需要2^16=65536个条目
            if self.y_out_type == 0:  # 输出INT8
                self.lut_in_len = 65536 // 32  # 65536个INT8值，每32B存储32个INT8
            else:  # 输出BF16
                self.lut_in_len = 65536 // 16  # 65536个BF16值，每32B存储16个BF16
        
        # 保存原语需要提前配置的数据，如果有para则必须在此生成
        # 对于非para的数据，可以有3种情况：
        # 1. 为None，表示不需要提前配置，可能是其他原语的输出作为输入，直接跳过这份数据即可
        # 2. 为[]，表示需要提前配置，需要生成一个DataBlock，长度根据原语参数确定，zero=0表示全随机
        # 3. 为DataBlock，表示直接使用这个DataBlock 
        if x_in is not None:  # 如果x_in为None，表示不需要提前配置，不需要处理
            if x_in == []:
                self.data_blocks["x_in"] = DataBlock(data=None, length=self.x_in_len, zero=0, addressing="32B")
            else:
                self.data_blocks["x_in"] = deepcopy(x_in)
        
        if lut_in is not None:
            if lut_in == []:
                self.data_blocks["lut_in"] = DataBlock(data=None, length=self.lut_in_len, zero=0, addressing="32B")
            else:
                self.data_blocks["lut_in"] = deepcopy(lut_in)

        
        # 计算地址 起始地址和数据长度
        self.setAddrBlocks()
        
        # 生成PIC
        self.setPIC()
        
    def setPIC(self):
        self.PIC = intbv(0, min=0, max=(1<<256))
        
        # 根据表格更新bit位置
        self.PIC[8:0] = 0xF3        # PIC(Code): [0:7]
        self.PIC[16:8] = self.deps   # deps: [8:15]
        self.PIC[47:32] = self.x_in_addr  # x_in_addr: [32:46] - 15bit
        self.PIC[63:48] = self.lut_in_addr  # lut_in_addr: [48:62] - 15bit
        self.PIC[79:64] = self.y_out_addr   # y_out_addr: [64:78] - 15bit
        self.PIC[96:80] = self.vector_num   # vector_num: [80:95]
        self.PIC[112:96] = self.vector_len_in_32B  # vector_len_in_32B: [96:111]
        self.PIC[130] = self.x_in_type  # x_in_type: [130]
        self.PIC[131] = self.y_out_type # y_out_type: [131]
        
    def setAddrBlocks(self):
        # 除了各种输入数据外，还需要记录结果数据的地址、Para的地址等各种运行中需要的数据地址核长度
        # 此函数不应该直接使用self.data_blocks中的数据长度，因为这里计算的长度需要和self.data_blocks中的数据长度进行对比以确保传入的数据长度是正确的
        # self.data_addr_list["data_name"] = AddrBlocksInMem({addr: length}, addressing)

        self.data_addr_list["x_in"] = AddrBlocksInMem({self.x_in_addr: self.x_in_len}, "32B")
        self.data_addr_list["lut_in"] = AddrBlocksInMem({self.lut_in_addr: self.lut_in_len}, "32B")
        self.data_addr_list["y_out"] = AddrBlocksInMem({self.y_out_addr: self.y_out_len}, "32B")
    
    def execute(self, core: EmuCore):
        # Data type: 0 = INT8, 1 = BF16
        # Determine configuration based on input type
        if self.x_in_torch_dtype == torch.int8:  # INT8: 32 elements per 32B
            in_elements_per_cell = 32
            lut_entries = 256  # 2^8
        elif self.x_in_torch_dtype == torch.bfloat16:  # BF16: 16 elements per 32B
            in_elements_per_cell = 16
            lut_entries = 65536  # 2^16
        
        # Read input tensor
        input_tensor_size = (self.vector_num + 1, (self.vector_len_in_32B + 1) * in_elements_per_cell)
        input_tensor = core.memory.readTensor(
            self.x_in_addr, self.x_in_len, input_tensor_size, self.x_in_torch_dtype, (0, 1)
        )
        
        # Read LUT table
        lut_tensor_size = (lut_entries,)
        lut_tensor = core.memory.readTensor(
            self.lut_in_addr, self.lut_in_len, lut_tensor_size, self.y_out_torch_dtype, (0,)
        )
        
        # Convert input to indices
        if self.x_in_torch_dtype == torch.int8:  # INT8 -> reinterpret as uint8 [0, 255]
            indices = input_tensor.view(torch.uint8).to(torch.long)
        else:  # BF16 -> reinterpret as uint16 [0, 65535]
            indices = input_tensor.view(torch.uint16).to(torch.long)
        
        # Use indices to look up LUT table
        output = lut_tensor[indices]
        
        # Write the output tensor to memory
        core.memory.writeTensor(self.y_out_addr, output, (0, 1))
    
    def generate_events(self, core: AnalyCore):
        events = []
        elements_per_vector = (self.vector_len_in_32B + 1) * get_elements_num_in_cell(self.x_in_torch_dtype)
        total_elements = (self.vector_num + 1) * elements_per_vector
        computation = total_elements
        
        if self.x_in_torch_dtype == self.y_out_torch_dtype:  # INT8 -> INT8 or BF16 -> BF16
            cycles_per_cell = 4 + get_elements_num_in_cell(self.x_in_torch_dtype) * 2 + 1 # read + n * (lookup + select) + write = 2 + n * (1 + 1) + 1 = 4 cycles per cell, where n is the number of elements in one input cell (32 for INT8, 16 for BF16)
            cycles_per_vector = (self.vector_len_in_32B + 1) * cycles_per_cell
        elif self.x_in_torch_dtype == torch.int8 and self.y_out_torch_dtype == torch.bfloat16:  # INT8 -> BF16
            cycles_per_cell = 69 # read + 16*(lookup+select) + buffer + 16*(lookup + select) + write_buffer + write = 2 + 16*(1+1) + 1 + 16*(1+1) + 1 + 1 = 69 cycles per cell
            cycles_per_vector = (self.vector_len_in_32B + 1) * cycles_per_cell
        elif self.x_in_torch_dtype == torch.bfloat16 and self.y_out_torch_dtype == torch.int8:  # BF16 -> INT8
            cycles_per_cell_without_write = 34 # read + 16*(lookup+select) = 2 + 16*(1+1) = 34 cycles per cell
            write_cycles_per_vector = np.ceil(elements_per_vector / 32) # every 32 output elements (32 INT8 values) need 1 cycle to write
            cycles_per_vector = (self.vector_len_in_32B + 1) * cycles_per_cell_without_write + write_cycles_per_vector

        total_cycle = (self.vector_num + 1) * cycles_per_vector
        
        compute_event = ComputeEvent(
            name="Lut",
            parent=core.vector.full_name,
            compute_type=EventType.VECTOR,
            computation=computation,
            theoretical_computation=(self.vector_num + 1) * elements_per_vector,
            max_consume_rate=computation / total_cycle,
            energy=0, # consider the energy of LUT as 0
        )
        events.append(compute_event)
        
        # read
        L0_input_volume = total_elements * get_byte_num(self.x_in_torch_dtype) # the volume of input data read from L0 memory, in bytes
        L0_lookup_volume = total_elements * 32 # the volume of LUT data read from L0 memory, in bytes. Each input element requires one LUT lookup which is one cell(32B).
        L0_read_volume = L0_input_volume + L0_lookup_volume

        L0_read_event = MemoryEvent(
            name="Lut_read_L0",
            parent=core.memory[0].full_name,
            memory_type=EventType.READ,
            volume=L0_read_volume,
            bounded_events=[],
            energy=np.ceil(L0_read_volume / core.config["core"]["memory_width"]) * core.config["core"]["L0_memory_read_energy"], 
            max_bandwidth=L0_read_volume / total_cycle,
            hierarchy=0
        )
        events.append(L0_read_event)

        # write
        total_output_cells = (self.vector_num + 1) * np.ceil(elements_per_vector / get_elements_num_in_cell(self.y_out_torch_dtype))
        L0_write_volume = total_output_cells * 32 # the volume of data written to L0 memory, in bytes
        
        L0_write_event = MemoryEvent(
            name="Lut_write_L0",
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