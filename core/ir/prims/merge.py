from core.ir.prims.prim import Primitive
from basics.addr_blocks_in_mem import AddrBlocksInMem
from basics.data_block import DataBlock
from core.simulator.emulator.core import Core as EmuCore
from core.simulator.analyser.core import Core as AnalyCore
from core.simulator.analyser.event import EventType, ComputeEvent, MemoryEvent
import torch
import math
from myhdl import bin, intbv

class PrimMerge(Primitive):
    def __init__(self, deps, data_type, input1_addr, input2_addr, output_addr, 
                input1_num, input2_num, output_num, input1_len_cell, input2_len_cell, output_len_cell):
        super().__init__()
        self.name = "Merge"
        self.deps = deps
        self.data_type = data_type
        self.input1_addr = input1_addr
        self.input2_addr = input2_addr
        self.output_addr = output_addr
        self.input1_num = input1_num
        self.input2_num = input2_num
        self.output_num = output_num
        self.input1_len_cell = input1_len_cell
        self.input2_len_cell = input2_len_cell
        self.output_len_cell = output_len_cell
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
            raise ValueError("Unsupported data type for Merge operation.")

        # 生成PIC
        self.setPIC()
        
    def setPIC(self):
        self.PIC = intbv(0, min=0, max=(1<<256))
        self.PIC[4:0] = 0x2
        self.PIC[8:4] = 0x0
        self.PIC[16:8] = self.deps
        self.PIC[31:16] = self.input1_addr
        self.PIC[48:32] = self.input1_num
        self.PIC[64:48] = self.input1_len_cell
        self.PIC[79:64] = self.input2_addr
        self.PIC[96:80] = self.input2_num
        self.PIC[112:96] = self.input2_len_cell
        self.PIC[127:112] = self.output_addr
        self.PIC[144:128] = self.output_num
        self.PIC[160:144] = self.output_len_cell
        
    def setAddrBlocks(self):
        pass

    def execute(self, core: EmuCore):
        # Get the input tensor
        input1_data_length = core.memory.getTensorLen((self.input1_num, self.input1_len_cell * self.per_cell_elements), self.tensor_data_type, (0, 1))
        input1_tensor = core.memory[self.input1_addr:self.input1_addr + input1_data_length].view(self.tensor_data_type).reshape((self.input1_num, -1))[:, :(self.input1_len_cell * self.per_cell_elements)].permute(0, 1)
        input2_data_length = core.memory.getTensorLen((self.input2_num, self.input2_len_cell * self.per_cell_elements), self.tensor_data_type, (0, 1))
        input2_tensor = core.memory[self.input2_addr:self.input2_addr + input2_data_length].view(self.tensor_data_type).reshape((self.input2_num, -1))[:, :(self.input2_len_cell * self.per_cell_elements)].permute(0, 1)           
        
        # Force type transformation
        input1_tensor = input1_tensor.to(self.tensor_data_type)
        input2_tensor = input2_tensor.to(self.tensor_data_type)
        
        # Merge the two input tensors into one output tensor
        # len dimension pad or cut
        if self.output_len_cell < self.input1_len_cell:
            input1_tensor = input1_tensor[:, 0:self.output_len_cell*self.per_cell_elements]
        elif self.output_len_cell < self.input1_len_cell + self.input2_len_cell:
            input2_tensor = input2_tensor[:, 0:(self.output_len_cell - self.input1_len_cell)*self.per_cell_elements]
        elif self.output_len_cell > self.input1_len_cell + self.input2_len_cell:
            input2_tensor = torch.cat((input2_tensor, torch.full((self.input2_num, (self.output_len_cell - self.input1_len_cell - self.input2_len_cell)*self.per_cell_elements) ,0, dtype=self.tensor_data_type)), dim=1)
        # num dimension pad or cut
        if self.output_num < self.input1_num:
            input1_tensor = input1_tensor[0:self.output_num, :]
        if self.output_num < self.input2_num:
            input2_tensor = input2_tensor[0:self.output_num, :]
        if self.output_num > self.input1_num:
            input1_tensor = torch.cat((input1_tensor, torch.full((self.output_num - self.input1_num, input1_tensor.size(1)) ,0, dtype=self.tensor_data_type)), dim=0)
        if self.output_num > self.input2_num:
            input2_tensor = torch.cat((input2_tensor, torch.full((self.output_num - self.input2_num, input2_tensor.size(1)) ,0, dtype=self.tensor_data_type)), dim=0)
        
        # Calculate output tensor
        if self.output_len_cell <= self.input1_len_cell:
            output_tensor = input1_tensor
        else:
            output_tensor = torch.cat((input1_tensor, input2_tensor), dim=1)
        
        # Write the output tensor to the memory
        core.memory.writeTensor(self.output_addr, output_tensor, (0, 1))

    def generate_events(self, core: AnalyCore):
        events = []
        # merge不涉及计算，只涉及数据搬运
        # 这里 input1 和 input2 可能存在部分无需读取
        # input1 实际读取长度
        input1_real_num = min(self.output_num, self.input1_num)
        input1_real_len = min(self.output_len_cell, self.input1_len_cell)
        # input2 实际读取长度
        input2_real_num = min(self.output_num, self.input2_num)
        input2_real_len = max(0, self.output_len_cell - self.input1_len_cell)

        # 暂时只考虑单级缓存
        
        # 总的读写时钟数 read + write (read需要两个时钟完成，write只需要一个时钟完成)
        total_cycle = (input1_real_len * input1_real_num + input2_real_len * input2_real_num) * 2 + (self.output_len_cell * self.output_num)
        
        # read
        L0_volume = 0
        
        # 计算容量要求(单位字节：Byte)
        # 这里 input1 和 input2 可能存在部分无需读取
        # input1
        # len * num * parallel_elements * bytes_per_element
        L0_volume += input1_real_len * input1_real_num * self.per_cell_elements * (32 / self.per_cell_elements)
        # input2
        # len * num * parallel_elements * bytes_per_element
        L0_volume += input2_real_len * input2_real_num * self.per_cell_elements * (32 / self.per_cell_elements)

        # read event
        L0_read_event = MemoryEvent(
            name="Merge_read_L0",
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
        output_volume = self.output_len_cell * self.output_num * self.per_cell_elements * (32 / self.per_cell_elements)
        # write event
        output_event = MemoryEvent(
            name="Merge_write",
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
