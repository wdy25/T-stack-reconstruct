from core.ir.prims.prim import Primitive
from basics.addr_blocks_in_mem import AddrBlocksInMem
from basics.data_block import DataBlock
from core.simulator.emulator.core import Core as EmuCore
from core.simulator.analyser.core import Core as AnalyCore
from core.simulator.analyser.event import EventType, ComputeEvent, MemoryEvent
import torch
import math
from myhdl import bin, intbv

class PrimSplit(Primitive):
    def __init__(self, deps, data_type, input_addr, output1_addr, output2_addr, 
                input_num, output1_num, output2_num, 
                input_len_cell, output1_len_cell, output2_len_cell):
        super().__init__()
        self.name = "Split"
        self.deps = deps
        self.data_type = data_type
        self.input_addr = input_addr
        self.output1_addr = output1_addr
        self.output2_addr = output2_addr
        self.input_num = input_num
        self.output1_num = output1_num
        self.output2_num = output2_num
        self.input_len_cell = input_len_cell
        self.output1_len_cell = output1_len_cell
        self.output2_len_cell = output2_len_cell
        
        # data_type: 0 -> SPIKE, 1 -> INT8, 2 -> BF16
        if self.data_type == 0:
            self.tensor_data_type = torch.bool
            self.per_cell_elements = 256
        elif self.data_type == 1:
            self.tensor_data_type = torch.int8
            self.per_cell_elements = 32
        elif self.data_type == 2:
            self.tensor_data_type = torch.bfloat16
            self.per_cell_elements = 16
        else:
            raise ValueError("Unsupported data type for Split operation.")

        # 生成PIC
        self.setPIC()
        
    def setPIC(self):
        self.PIC = intbv(0, min=0, max=(1<<256))
        self.PIC[4:0] = 0x2
        self.PIC[8:4] = 0x1
        self.PIC[16:8] = self.deps
        self.PIC[31:16] = self.input_addr
        self.PIC[48:32] = self.input_num
        self.PIC[64:48] = self.input_len_cell
        self.PIC[79:64] = self.output1_addr
        self.PIC[96:80] = self.output1_num
        self.PIC[112:96] = self.output1_len_cell
        self.PIC[127:112] = self.output2_addr
        self.PIC[144:128] = self.output2_num
        self.PIC[160:144] = self.output2_len_cell
        
    def setAddrBlocks(self):
        pass

    def execute(self, core: EmuCore):
        # Get the input tensor
        
        input_data_length = core.memory.getTensorLen((self.input_num, self.input_len_cell * self.per_cell_elements), self.tensor_data_type, (0, 1))
        input_tensor = core.memory[self.input_addr:self.input_addr + input_data_length].view(self.tensor_data_type).reshape((self.input_num, -1))[:, :(self.input_len_cell * self.per_cell_elements)].permute(0, 1)
        
        # Force type transformation
        input_tensor = input_tensor.to(self.tensor_data_type)
        
        # Split the input tensor into two output tensors
        max_num = max(self.output1_num, self.output2_num)
        # len dimension pad or cut
        if self.input_len_cell < self.output1_len_cell + self.output2_len_cell:
            input_tensor = torch.cat((input_tensor, torch.full((self.input_num, (self.output1_len_cell + self.output2_len_cell - self.input_len_cell)*self.per_cell_elements) ,0 , dtype=self.tensor_data_type)), dim=1)
        elif self.input_len_cell > self.output1_len_cell + self.output2_len_cell:
            input_tensor = input_tensor[:, 0:(self.output1_len_cell + self.output2_len_cell - self.input_len_cell)*self.per_cell_elements]
        # num dimension pad or cut
        if max_num > self.input_num:
            input_tensor = torch.cat((input_tensor, torch.full((max_num - self.input_num, input_tensor.size(1)) ,0, dtype=self.tensor_data_type)), dim=0)
            print(input_tensor.shape)
        
        # Calculate output tensors
        output1_tensor = input_tensor[:self.output1_num, :(self.output1_len_cell*self.per_cell_elements)]
        output2_tensor = input_tensor[:self.output2_num, (self.output1_len_cell*self.per_cell_elements):((self.output1_len_cell+self.output2_len_cell)*self.per_cell_elements)]
        
        # Write the output tensor to the memory
        core.memory.writeTensor(self.output1_addr, output1_tensor, (0, 1))
        core.memory.writeTensor(self.output2_addr, output2_tensor, (0, 1))

    def generate_events(self, core: AnalyCore):
        events = []
        # split不涉及计算，只涉及数据搬运
        # 这里 input 可能存在部分无需读取
        # input 实际读取长度
        input_real_num = min(self.output1_num, self.output2_num, self.input_num)
        input_real_len = min(self.output1_len_cell + self.output2_len_cell, self.input_len_cell)

        # 暂时只考虑单级缓存
        # input_hierarchy = 0 if self.x_in_addr < core.config["core"]["L0_memory_capacity"] // core.config["core"]["memory_width"] else 1
        # output_hierarchy = 0 if self.y_out_addr < core.config["core"]["L0_memory_capacity"] // core.config["core"]["memory_width"] else 1

        # 总的读写时钟数 read + write (read需要两个时钟完成，write只需要一个时钟完成)
        total_cycle = input_real_len * input_real_num * 2 + (self.output1_len_cell * self.output1_num + self.output2_len_cell * self.output2_num)
        
        # read
        L0_volume = 0
        
        # 计算容量要求(单位字节：Byte)
        # 这里 input 可能存在部分无需读取
        # input
        # len * num * parallel_elements * bytes_per_element
        L0_volume += input_real_len * input_real_num * self.per_cell_elements * (32 / self.per_cell_elements)

        # read event
        L0_read_event = MemoryEvent(
            name="Split_read_L0",
            # 这个parent是啥？
            parent=core.memory[0].full_name,
            memory_type=EventType.READ,
            volume=L0_volume,
            bounded_events=[],
            energy=math.ceil(L0_volume / core.config["core"]["memory_width"]) * core.config["core"]["L0_memory_read_energy"], 
            max_bandwidth=L0_volume / total_cycle,
            hierarchy=0
        )
        events.append(L0_read_event)

        # 计算容量要求(单位字节：Byte)
        # len * num * parallel_elements * bytes_per_element
        output_volume = self.output1_len_cell * self.output1_num * self.per_cell_elements * (32 / self.per_cell_elements) + self.output2_len_cell * self.output2_num * self.per_cell_elements * (32 / self.per_cell_elements)
        # write event
        output_event = MemoryEvent(
            name="Split_write",
            # 这个parent是啥？原本这里的索引是hierarchy
            parent=core.memory[0].full_name,
            memory_type=EventType.WRITE,
            volume=output_volume,
            bounded_events=[],
            energy=(math.ceil(output_volume / core.config["core"]["memory_width"]) * core.config["core"]["L0_memory_write_energy"]), 
            max_bandwidth=output_volume / total_cycle,
            hierarchy=0
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