import torch

from copy import deepcopy
from myhdl import bin, intbv

from core.ir.prims.prim import Primitive
from basics.addr_blocks_in_mem import AddrBlocksInMem
from basics.data_block import DataBlock
from core.simulator.emulator.core import Core as EmuCore
from core.simulator.analyser.core import Core as AnalyCore
from core.simulator.analyser.event import EventType, ComputeEvent, MemoryEvent
from core.utils.get_byte_num import get_torch_dtype_from_type_num, get_elements_num_in_cell, get_byte_num
import numpy as np
import math

class PrimConvert(Primitive):
    def __init__(self, deps, x_in_addr, y_out_addr, 
                 vector_num, vector_len_in_32B, x_in_type, y_out_type, 
                 x_in: DataBlock=None):
        super().__init__()
        self.name = "Convert"  # 定义名称
        
        # 记录参数，即原语中的所有字段，包括para中某些需要额外配置的字段
        self.deps = deps
        self.x_in_addr = x_in_addr
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
        # 0: INT8, 1: BF16, 2: BIN
        if self.x_in_type == self.y_out_type:  # 相同类型
            self.y_out_len = (self.vector_num+1) * (self.vector_len_in_32B+1)
        elif self.x_in_type == 0 and self.y_out_type == 1:  # INT8 -> BF16
            self.y_out_len = (self.vector_num+1) * (self.vector_len_in_32B+1) * 2
        elif self.x_in_type == 1 and self.y_out_type == 0:  # BF16 -> INT8
            self.y_out_len = (self.vector_num+1) * math.ceil((self.vector_len_in_32B+1) / 2)
        elif self.x_in_type == 0 and self.y_out_type == 2:  # INT8 -> BIN
            # 1个INT8(8bit) -> 1个BIN(1bit)，8个INT8占1个cell -> 8个BIN占1个cell，所以长度缩小8倍
            self.y_out_len = (self.vector_num+1) * math.ceil((self.vector_len_in_32B+1) / 8)
        elif self.x_in_type == 1 and self.y_out_type == 2:  # BF16 -> BIN  
            # 1个BF16(16bit) -> 1个BIN(1bit)，16个BF16占1个cell -> 16个BIN占1个cell，所以长度缩小16倍
            self.y_out_len = (self.vector_num+1) * math.ceil((self.vector_len_in_32B+1) / 16)
        elif self.x_in_type == 2 and self.y_out_type == 0:  # BIN -> INT8
            # 1个BIN(1bit) -> 1个INT8(8bit)，8个BIN占1个cell -> 8个INT8占1个cell，所以长度扩大8倍
            self.y_out_len = (self.vector_num+1) * (self.vector_len_in_32B+1) * 8
        elif self.x_in_type == 2 and self.y_out_type == 1:  # BIN -> BF16
            # 1个BIN(1bit) -> 1个BF16(16bit)，16个BIN占1个cell -> 16个BF16占1个cell，所以长度扩大16倍
            self.y_out_len = (self.vector_num+1) * (self.vector_len_in_32B+1) * 16
        else:
            # 默认情况，相同长度
            self.y_out_len = (self.vector_num+1) * (self.vector_len_in_32B+1)
        
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
        
        # 计算地址 起始地址和数据长度
        self.setAddrBlocks()
        
        # 生成PIC
        self.setPIC()
        
    def setPIC(self):
        self.PIC = intbv(0, min=0, max=(1<<256))
        
        # 根据表格更新bit位置
        self.PIC[8:0] = 0x33        # PIC(Code): [0:7]
        self.PIC[16:8] = self.deps   # deps: [8:15]
        self.PIC[47:32] = self.x_in_addr  # x_in_addr: [32:46] - 15bit
        self.PIC[79:64] = self.y_out_addr   # y_out_addr: [64:78] - 15bit
        self.PIC[96:80] = self.vector_num   # vector_num: [80:95]
        self.PIC[112:96] = self.vector_len_in_32B  # vector_len_in_32B: [96:111]
        self.PIC[130:128] = self.x_in_type  # x_in_type: [128:129] - 2bit
        self.PIC[132:130] = self.y_out_type # y_out_type: [130:131] - 2bit
        
    def setAddrBlocks(self):
        # 除了各种输入数据外，还需要记录结果数据的地址、Para的地址等各种运行中需要的数据地址核长度
        # 此函数不应该直接使用self.data_blocks中的数据长度，因为这里计算的长度需要和self.data_blocks中的数据长度进行对比以确保传入的数据长度是正确的
        # self.data_addr_list["data_name"] = AddrBlocksInMem({addr: length}, addressing)

        self.data_addr_list["x_in"] = AddrBlocksInMem({self.x_in_addr: self.x_in_len}, "32B")
        self.data_addr_list["y_out"] = AddrBlocksInMem({self.y_out_addr: self.y_out_len}, "32B")
    
    def execute(self, core: EmuCore):
        # Data type: 0 = INT8, 1 = BF16, 2 = BIN (bool)
        x_in_type = int(self.x_in_type)
        y_out_type = int(self.y_out_type)
        
        # Determine elements per 32B cell and read configuration based on input type
        if x_in_type == 0:  # INT8: 32 elements per 32B
            elements_per_cell = 32
            in_dtype = torch.int8
        elif x_in_type == 1:  # BF16: 16 elements per 32B
            elements_per_cell = 16
            in_dtype = torch.bfloat16
        else:  # BIN (bool): 256 elements per 32B (256 bits)
            elements_per_cell = 256
            in_dtype = torch.bool
        
        # Read input tensor
        input_tensor_size = (self.vector_num + 1, (self.vector_len_in_32B + 1) * elements_per_cell)
        input_tensor = core.memory.readTensor(
            self.x_in_addr, self.x_in_len, input_tensor_size, in_dtype, (0, 1)
        )
        
        # Perform type conversion
        if x_in_type == y_out_type:
            # Same type, direct copy
            output = input_tensor
        elif x_in_type == 0 and y_out_type == 1:  # INT8 -> BF16
            output = input_tensor.to(torch.bfloat16)
        elif x_in_type == 1 and y_out_type == 0:  # BF16 -> INT8
            output = input_tensor.clamp(-128, 127).to(torch.int8)
        elif x_in_type == 0 and y_out_type == 2:  # INT8 -> BIN
            # Positive intger (excluding 0) -> 1, Negative integer(including 0) -> 0
            output = (input_tensor > 0).to(torch.bool)
        elif x_in_type == 1 and y_out_type == 2:  # BF16 -> BIN
            # Positive (excluding +0) -> 1, negative (including +0 and -0) -> 0
            # Reinterpret BF16 as int16 to find positive/negative easilier
            bf16_as_int = input_tensor.view(torch.int16)
            # bf16_as_int > 0 means positive, else means negative (including +0 and -0)
            output = (bf16_as_int > 0).to(torch.bool)
        elif x_in_type == 2 and y_out_type == 0:  # BIN -> INT8
            # BIN=0 -> 0, BIN=1 -> 1
            output = input_tensor.to(torch.int8)
        elif x_in_type == 2 and y_out_type == 1:  # BIN -> BF16
            # BIN=0 -> -0.0, BIN=1 -> 1.0
            output = torch.where(
                input_tensor,
                torch.tensor(1.0, dtype=torch.bfloat16),
                torch.tensor(-0.0, dtype=torch.bfloat16)  # negative zero
            )
        else:
            # Warning: Unsupported type conversion
            raise ValueError(f"Unsupported type conversion from {x_in_type} to {y_out_type}.")
        
        # Write the output tensor to memory
        core.memory.writeTensor(self.y_out_addr, output, (0, 1))
    
    def generate_events(self, core: AnalyCore):
        events = []
        elements_per_vector = (self.vector_len_in_32B + 1) * get_elements_num_in_cell(self.x_in_torch_dtype)
        total_elements = (self.vector_num + 1) * elements_per_vector
        computation = total_elements
        
        if self.x_in_torch_dtype == torch.int8 and self.y_out_torch_dtype == torch.bfloat16:  # INT8 -> BF16
            cycles_per_cell = 6 # read + convert + write + convert + write = 2 + 1 + 1 + 1 + 1 = 6 cycles per cell
            cycles_per_vector = (self.vector_len_in_32B + 1) * cycles_per_cell
        elif self.x_in_torch_dtype in (torch.int8, torch.bfloat16) and self.y_out_torch_dtype == torch.bool:  # INT8/BF16 -> BIN
            cycles_per_cell_without_write = 3 # read + convert = 2 + 1 = 3 cycles per cell
            write_cycles_per_vector = np.ceil(elements_per_vector / 256) # each cell can write 256 BIN elements, so need to divide by 256 and round up
            cycles_per_vector = (self.vector_len_in_32B + 1) * cycles_per_cell_without_write + write_cycles_per_vector
        elif self.x_in_torch_dtype == torch.bfloat16 and self.y_out_torch_dtype == torch.int8:  # BF16 -> INT8
            cycles_per_cell_without_write = 3 # read + convert = 2 + 1 = 3 cycles per cell
            write_cycles_per_vector = np.ceil(elements_per_vector / 32) # each cell can write 32 INT8 elements, so need to divide by 32 and round up
            cycles_per_vector = (self.vector_len_in_32B + 1) * cycles_per_cell_without_write + write_cycles_per_vector
        elif self.x_in_torch_dtype == torch.bool and self.y_out_torch_dtype in (torch.int8, torch.bfloat16):  # BIN -> INT8/BF16
            converted_cells_num = np.ceil(elements_per_vector / get_elements_num_in_cell(self.y_out_torch_dtype)) # number of cells after conversion
            cycles_per_cell = 2 + converted_cells_num * 2 # read + n * (convert + write) = 2 + n * (1 + 1), where n is the number of cells that one BIN cell can be converted into for INT8/BF16 (8 for INT8, 16 for BF16)
            cycles_per_vector = (self.vector_len_in_32B + 1) * cycles_per_cell
            
        total_cycle = (self.vector_num + 1) * cycles_per_vector
        
        compute_event = ComputeEvent(
            name="Convert",
            parent=core.vector.full_name,
            compute_type=EventType.VECTOR,
            computation=computation,
            theoretical_computation=(self.vector_num + 1) * elements_per_vector,
            max_consume_rate=computation / total_cycle,
            energy=computation * core.config["core"]["int8_PE_add_energy"],
        )
        events.append(compute_event)
        
        # read
        L0_volume = total_elements * get_byte_num(self.x_in_torch_dtype) # the volume of data read from L0, in bytes
            
        L0_read_event = MemoryEvent(
            name="Convert_read_L0",
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
            name="Convert_write_L0",
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
