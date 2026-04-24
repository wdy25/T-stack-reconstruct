from copy import deepcopy
from myhdl import bin, intbv
import torch
import numpy as np

from core.ir.prims.prim import Primitive
from basics.addr_blocks_in_mem import AddrBlocksInMem
from basics.data_block import DataBlock
from core.simulator.emulator.core import Core as EmuCore
from core.simulator.analyser.core import Core as AnalyCore
from core.simulator.analyser.event import EventType, ComputeEvent, MemoryEvent
from core.utils.get_byte_num import get_elements_num_in_cell, get_byte_num

def to_signed_bits(val, width=16):
    if not -(1 << (width-1)) <= val < (1 << (width-1)):
        raise ValueError("超出范围: %d位有符号数" % width)
    return val & ((1 << width) - 1)  # 得到补码

class PrimTransposition(Primitive):
    # weight: [kh(A),kw(B),cin(C),cout(D)]
    def __init__(self,
        transpose_type,
        transpose_order,
        deps,
        dim_A,
        dim_B,
        dim_C,
        dim_D,
        x_in_addr,
        y_out_addr,
        param_addr_1,
        # x_in_data
        ):
        super().__init__()
        self.name = "Transposition"
        
        self.transpose_type = transpose_type
        self.transpose_order = transpose_order
        if self.transpose_type == 'INT8':
            self.x_in_torch_dtype = torch.int8
            self.w_dw = 8
            self.reg_depth = 32
            self.data_type = torch.int8
        elif self.transpose_type == 'BF16':
            self.x_in_torch_dtype = torch.bfloat16
            self.w_dw = 16
            self.reg_depth = 16
            self.data_type = torch.bfloat16
        self.array_row = 256 / self.w_dw
        self.deps = deps
        self.dim_A = dim_A
        self.dim_B = dim_B
        self.dim_C = dim_C
        self.dim_D = dim_D
        self.x_in_addr = x_in_addr
        self.y_out_addr = y_out_addr
        self.param_addr_1 = param_addr_1
        if self.transpose_type == 'INT8' and self.transpose_order == 'AB':
            self.dim = 0b1001
        if self.transpose_type == 'INT8' and self.transpose_order == 'AC':
            self.dim = 0b1010
        if self.transpose_type == 'INT8' and self.transpose_order == 'BC':
            self.dim = 0b1011
        if self.transpose_type == 'INT8' and self.transpose_order == 'AD':
            self.dim = 0b1101
            self.dim_head = self.dim_A // self.array_row
            self.dim_res = self.dim_A % self.array_row
            self.dim_new = self.dim_A if self.dim_res == 0 else (self.dim_head + 1) * self.array_row
        if self.transpose_type == 'INT8' and self.transpose_order == 'BD':
            self.dim = 0b1110
            self.dim_head = self.dim_B // self.array_row
            self.dim_res = self.dim_B % self.array_row
            self.dim_new = self.dim_B if self.dim_res == 0 else (self.dim_head + 1) * self.array_row
        if self.transpose_type == 'INT8' and self.transpose_order == 'CD':
            self.dim = 0b1111
            self.dim_head = self.dim_C // self.array_row
            self.dim_res = self.dim_C % self.array_row
            self.dim_new = self.dim_C if self.dim_res == 0 else (self.dim_head + 1) * self.array_row
        if self.transpose_type == 'BF16' and self.transpose_order == 'AB':
            self.dim = 0b0001
        if self.transpose_type == 'BF16' and self.transpose_order == 'AC':
            self.dim = 0b0010
        if self.transpose_type == 'BF16' and self.transpose_order == 'BC':
            self.dim = 0b0011
        if self.transpose_type == 'BF16' and self.transpose_order == 'AD':
            self.dim = 0b0101
            self.dim_res = self.dim_A % self.array_row
            if self.dim_res == 0:
                self.dim_head = self.dim_A // self.array_row - 1
            else:
                self.dim_head = self.dim_A // self.array_row
            self.dim_new = self.dim_A if self.dim_res == 0 else (self.dim_head + 1) * self.array_row
        if self.transpose_type == 'BF16' and self.transpose_order == 'BD':
            self.dim = 0b0110
            self.dim_res = self.dim_B % self.array_row
            if self.dim_res == 0:
                self.dim_head = self.dim_B // self.array_row - 1
            else:
                self.dim_head = self.dim_B // self.array_row
            self.dim_new = self.dim_B if self.dim_res == 0 else (self.dim_head + 1) * self.array_row
        if self.transpose_type == 'BF16' and self.transpose_order == 'CD':
            self.dim = 0b0111
            self.dim_res = self.dim_C % self.array_row
            if self.dim_res == 0:
                self.dim_head = self.dim_C // self.array_row - 1
            else:
                self.dim_head = self.dim_C // self.array_row
            self.dim_new = self.dim_C if self.dim_res == 0 else (self.dim_head + 1) * self.array_row

        # self.data_blocks["x_in_data"] = deepcopy(x_in_data)

        self.dim_D_head = self.dim_D // self.array_row
        self.dim_D_res = self.dim_D % self.array_row
        self.dim_D_new = self.dim_D if self.dim_D_res == 0 else (self.dim_D_head + 1) * self.array_row
        
        if self.transpose_order == 'AD' or self.transpose_order == 'BD' or self.transpose_order == 'CD':
            if self.transpose_order == 'AD':
                self.permute_order = (3,1,2,0)

                self.x_addr_gap_1 = self.dim_B * self.dim_C * self.dim_D_new * self.w_dw / 256
                self.x_addr_gap_2 = -(self.dim_A - 1) * self.dim_B * self.dim_C * self.dim_D_new * self.w_dw / 256 + self.dim_C * self.dim_D_new * self.w_dw / 256
                self.x_addr_gap_3 = -(self.dim_A - 1) * self.dim_B * self.dim_C * self.dim_D_new * self.w_dw / 256 - (self.dim_B - 1) * self.dim_C * self.dim_D_new * self.w_dw / 256 + self.dim_D_new * self.w_dw / 256
                self.x_addr_gap_4 = -(self.dim_A - 1) * self.dim_B * self.dim_C * self.dim_D_new * self.w_dw / 256 - (self.dim_B - 1) * self.dim_C * self.dim_D_new * self.w_dw / 256 - (self.dim_C - 1) * self.dim_D_new * self.w_dw / 256 + self.array_row * self.w_dw / 256

                self.w_addr_gap_1 = self.dim_B * self.dim_C * self.dim_new * self.w_dw / 256
                self.w_addr_gap_2 = -0 + self.dim_C * self.dim_new * self.w_dw / 256 - self.dim_head
                self.w_addr_gap_3 = -0 - (self.dim_B - 1) * self.dim_C * self.dim_new * self.w_dw / 256 - self.dim_head + self.dim_new * self.w_dw / 256
                self.w_addr_gap_4 = -0 - (self.dim_B - 1) * self.dim_C * self.dim_new * self.w_dw / 256 - self.dim_head - (self.dim_C - 1) * self.dim_new * self.w_dw / 256 + self.array_row * self.dim_B * self.dim_C * self.dim_new * self.w_dw / 256
            elif self.transpose_order == 'BD':
                self.permute_order = (0,3,2,1)
                
                self.x_addr_gap_1 = self.dim_C * self.dim_D_new * self.w_dw / 256
                self.x_addr_gap_2 = -(self.dim_B - 1) * self.dim_C * self.dim_D_new * self.w_dw / 256 + self.dim_B * self.dim_C * self.dim_D_new * self.w_dw / 256
                self.x_addr_gap_3 = -(self.dim_B - 1) * self.dim_C * self.dim_D_new * self.w_dw / 256 - (self.dim_A - 1) * self.dim_B * self.dim_C * self.dim_D_new * self.w_dw / 256 + self.dim_D_new * self.w_dw / 256
                self.x_addr_gap_4 = -(self.dim_B - 1) * self.dim_C * self.dim_D_new * self.w_dw / 256 - (self.dim_A - 1) * self.dim_B * self.dim_C * self.dim_D_new * self.w_dw / 256 - (self.dim_C - 1) * self.dim_D_new * self.w_dw / 256 + self.array_row * self.w_dw / 256

                self.w_addr_gap_1 = self.dim_C * self.dim_new * self.w_dw / 256
                self.w_addr_gap_2 = -0 - self.dim_head + self.dim_D * self.dim_C * self.dim_new * self.w_dw / 256
                self.w_addr_gap_3 = -0 - self.dim_head - (self.dim_A - 1) * self.dim_D * self.dim_C * self.dim_new * self.w_dw / 256 + self.dim_new * self.w_dw / 256
                self.w_addr_gap_4 = -0 - self.dim_head - (self.dim_A - 1) * self.dim_D * self.dim_C * self.dim_new * self.w_dw / 256 - (self.dim_C - 1) * self.dim_new * self.w_dw / 256 + self.array_row * self.dim_C * self.dim_new * self.w_dw / 256
            elif self.transpose_order == 'CD':
                self.permute_order = (0,1,3,2)

                self.x_addr_gap_1 = self.dim_D_new * self.w_dw / 256
                self.x_addr_gap_2 = -(self.dim_C - 1) * self.dim_D_new * self.w_dw / 256 + self.dim_B * self.dim_C * self.dim_D_new * self.w_dw / 256
                self.x_addr_gap_3 = -(self.dim_C - 1) * self.dim_D_new * self.w_dw / 256 - (self.dim_A - 1) * self.dim_B * self.dim_C * self.dim_D_new * self.w_dw / 256 + self.dim_C * self.dim_D_new * self.w_dw / 256
                self.x_addr_gap_4 = -(self.dim_C - 1) * self.dim_D_new * self.w_dw / 256 - (self.dim_A - 1) * self.dim_B * self.dim_C * self.dim_D_new * self.w_dw / 256 - (self.dim_B - 1) * self.dim_C * self.dim_D_new * self.w_dw / 256 + self.array_row * self.w_dw / 256

                self.w_addr_gap_1 = self.dim_new * self.w_dw / 256
                self.w_addr_gap_2 = -0 - self.dim_head + self.dim_B * self.dim_D * self.dim_new * self.w_dw / 256
                self.w_addr_gap_3 = -0 - self.dim_head - (self.dim_A - 1) * self.dim_B * self.dim_D * self.dim_new * self.w_dw / 256 + self.dim_D * self.dim_new * self.w_dw / 256
                self.w_addr_gap_4 = -0 - self.dim_head - (self.dim_A - 1) * self.dim_B * self.dim_D * self.dim_new * self.w_dw / 256 - (self.dim_B - 1) * self.dim_D * self.dim_new * self.w_dw / 256 + self.array_row * self.dim_new * self.w_dw / 256
        else:
            self.x_addr_gap_1 = 1
            self.w_addr_gap_1 = 0
            self.w_addr_gap_2 = 0
            self.w_addr_gap_3 = 0
            self.w_addr_gap_4 = 0
            if self.transpose_order == 'AB':
                self.permute_order = (1,0,2,3)

                self.x_addr_gap_2 = -(self.dim_D_new * self.w_dw / 256 - 1) * 1 + self.dim_D_new * self.w_dw / 256
                self.x_addr_gap_3 = -(self.dim_D_new * self.w_dw / 256 - 1) * 1 - (self.dim_C - 1) * self.dim_D_new * self.w_dw / 256 + self.dim_B * self.dim_C * self.dim_D_new * self.w_dw / 256
                self.x_addr_gap_4 = -(self.dim_D_new * self.w_dw / 256 - 1) * 1 - (self.dim_C - 1) * self.dim_D_new * self.w_dw / 256 - (self.dim_A - 1) * self.dim_B * self.dim_C * self.dim_D_new * self.w_dw / 256 + self.dim_C * self.dim_D_new * self.w_dw / 256
            elif self.transpose_order == 'AC':
                self.permute_order = (2,1,0,3)

                self.x_addr_gap_2 = -(self.dim_D_new * self.w_dw / 256 - 1) * 1 + self.dim_B * self.dim_C * self.dim_D_new * self.w_dw / 256
                self.x_addr_gap_3 = -(self.dim_D_new * self.w_dw / 256 - 1) * 1 - (self.dim_A - 1) * self.dim_B * self.dim_C * self.dim_D_new * self.w_dw / 256 + self.dim_C * self.dim_D_new * self.w_dw / 256
                self.x_addr_gap_4 = -(self.dim_D_new * self.w_dw / 256 - 1) * 1 - (self.dim_A - 1) * self.dim_B * self.dim_C * self.dim_D_new * self.w_dw / 256 - (self.dim_B - 1) * self.dim_C * self.dim_D_new * self.w_dw / 256 + self.dim_D_new * self.w_dw / 256
            elif self.transpose_order == 'BC':
                self.permute_order = (0,2,1,3)
                
                self.x_addr_gap_2 = -(self.dim_D_new * self.w_dw / 256 - 1) * 1 + self.dim_C * self.dim_D_new * self.w_dw / 256
                self.x_addr_gap_3 = -(self.dim_D_new * self.w_dw / 256 - 1) * 1 - (self.dim_B - 1) * self.dim_C * self.dim_D_new * self.w_dw / 256 + self.dim_D_new * self.w_dw / 256
                self.x_addr_gap_4 = -(self.dim_D_new * self.w_dw / 256 - 1) * 1 - (self.dim_B - 1) * self.dim_C * self.dim_D_new * self.w_dw / 256 - (self.dim_C - 1) * self.dim_D_new * self.w_dw / 256 + self.dim_B * self.dim_C * self.dim_D_new * self.w_dw / 256

        self.param1 = intbv(0)[256:]
        self.param1[16:0] = to_signed_bits(int(self.x_addr_gap_1),16)
        self.param1[32:16] = to_signed_bits(int(self.x_addr_gap_2),16)
        self.param1[48:32] = to_signed_bits(int(self.x_addr_gap_3),16)
        self.param1[64:48] = to_signed_bits(int(self.x_addr_gap_4),16)
        self.param1[80:64] = to_signed_bits(int(self.w_addr_gap_1),16)
        self.param1[96:80] = to_signed_bits(int(self.w_addr_gap_2),16)
        self.param1[112:96] = to_signed_bits(int(self.w_addr_gap_3),16)
        self.param1[128:112] = to_signed_bits(int(self.w_addr_gap_4),16)
        # self.data_blocks["param1"] = DataBlock(data=self.param1, length=1, zero=0, addressing="32B")

        # 计算地址
        self.setAddrBlocks()
        
        # 生成PIC
        self.setPIC()
        
    def setPIC(self):
        self.PIC = intbv(0, min=0, max=(1<<256))
        
        self.PIC[4:0] = 0x1
        self.PIC[8:4] = 0x6
        
        self.PIC[16:8] = self.deps
        self.PIC[32:16] = self.x_in_addr
        self.PIC[77:62] = self.y_out_addr
        self.PIC[92:77] = self.param_addr_1
        self.PIC[172:160] = self.dim_A - 1
        self.PIC[182:172] = self.dim_B - 1
        self.PIC[198:182] = self.dim_C - 1
        if (self.transpose_order == 'AD'):
            if (self.dim_A % self.array_row == 0):
                self.PIC[138:132] = self.dim_A // self.array_row - 1
            else:
                self.PIC[138:132] = self.dim_A // self.array_row
            self.PIC[144:138] = self.dim_A % self.array_row
        elif (self.transpose_order == 'BD'):
            if (self.dim_B % self.array_row == 0):
                self.PIC[138:132] = self.dim_B // self.array_row - 1
            else:
                self.PIC[138:132] = self.dim_B // self.array_row
            self.PIC[144:138] = self.dim_B % self.array_row
        elif (self.transpose_order == 'CD'):
            if (self.dim_C % self.array_row == 0):
                self.PIC[138:132] = self.dim_C // self.array_row - 1
            else:
                self.PIC[138:132] = self.dim_C // self.array_row
            self.PIC[144:138] = self.dim_C % self.array_row

        if (self.dim_D % self.reg_depth == 0):
            self.PIC[124:116] = self.dim_D // self.reg_depth - 1
        else:
            self.PIC[124:116] = self.dim_D // self.reg_depth
        self.PIC[132:124] = self.dim_D % self.reg_depth
        self.PIC[213:198] = self.dim_D - 1
        self.PIC[222:218] = self.dim

        
    def setAddrBlocks(self):
        pass
    
    def execute(self, core: EmuCore):
        # Determine element per cell
        element_per_cell = get_elements_num_in_cell(self.x_in_torch_dtype)
        
        # Calculate total input elements and memory cells
        tensor_shape = (self.dim_A, self.dim_B, self.dim_C, self.dim_D)
        total_elements = self.dim_A * self.dim_B * self.dim_C * self.dim_D
        input_cells = (total_elements + element_per_cell - 1) // element_per_cell
        
        # Read input tensor from memory
        x_in = core.memory.readTensor(
            self.x_in_addr,
            input_cells,
            tensor_shape,
            self.x_in_torch_dtype,
            (0, 1, 2, 3)
        )
        
        # Reshape to 4D tensor: (dim_A, dim_B, dim_C, dim_D)
        x_in = x_in.reshape(self.dim_A, self.dim_B, self.dim_C, self.dim_D)
        
        # Permute dimensions according to transpose_order
        if self.transpose_order == 'AB':
            # (A, B, C, D) -> (B, A, C, D)
            y_out = x_in.permute(1, 0, 2, 3)
        elif self.transpose_order == 'AC':
            # (A, B, C, D) -> (C, B, A, D)
            y_out = x_in.permute(2, 1, 0, 3)
        elif self.transpose_order == 'BC':
            # (A, B, C, D) -> (A, C, B, D)
            y_out = x_in.permute(0, 2, 1, 3)
        elif self.transpose_order == 'AD':
            # (A, B, C, D) -> (D, B, C, A)
            y_out = x_in.permute(3, 1, 2, 0)
        elif self.transpose_order == 'BD':
            # (A, B, C, D) -> (A, D, C, B)
            y_out = x_in.permute(0, 3, 2, 1)
        elif self.transpose_order == 'CD':
            # (A, B, C, D) -> (A, B, D, C)
            y_out = x_in.permute(0, 1, 3, 2)
        else:
            raise ValueError(f"Unknown transpose_order: {self.transpose_order}")
        
        # Ensure tensor is contiguous
        y_out = y_out.contiguous()
        
        # Write output tensor back to memory
        core.memory.writeTensor(self.y_out_addr, y_out, (0, 1, 2, 3))
    
    def generate_events(self, core: AnalyCore):
        events = []
        element_per_cell = get_elements_num_in_cell(self.x_in_torch_dtype)

        involves_dimD = self.transpose_order in ['AD', 'BD', 'CD']

        if involves_dimD:
            if self.transpose_order == 'AD':
                dim_x = self.dim_A
                outer = self.dim_B * self.dim_C
            elif self.transpose_order == 'BD':
                dim_x = self.dim_B
                outer = self.dim_A * self.dim_C
            else:  # CD
                dim_x = self.dim_C
                outer = self.dim_A * self.dim_B

            x_full_groups = dim_x // element_per_cell
            d_full_groups = self.dim_D // element_per_cell
            x_tail_cells = dim_x % element_per_cell
            d_tail_cells = self.dim_D % element_per_cell

            # Fixed modeling order for dimD-involved transpose:
            # outer loop: outer tiles, middle loop: D groups, inner loop: X groups.
            x_groups_per_outer = np.ceil(dim_x / element_per_cell)
            d_groups_per_outer = np.ceil(self.dim_D / element_per_cell)

            # For each outer tile, read path repeats the whole X traversal for every D group.
            total_read_cells = outer * dim_x * d_groups_per_outer
            # For each outer tile, write path outputs current D-group-sized slices for each X group.
            total_write_cells = outer * x_groups_per_outer * self.dim_D
            read_cycles_per_d_group = x_full_groups * (element_per_cell + 1) + (x_tail_cells + 1 if x_tail_cells > 0 else 0)
            write_cycles_per_x_group = d_full_groups * (element_per_cell + 1) + (d_tail_cells + 1 if d_tail_cells > 0 else 0)
            total_read_cycles = outer * d_groups_per_outer * read_cycles_per_d_group
            total_write_cycles = outer * x_groups_per_outer * write_cycles_per_x_group
        else:
            cells = self.dim_A * self.dim_B * self.dim_C * np.ceil(self.dim_D / element_per_cell)
            total_read_cells = cells
            total_write_cells = cells
            total_read_cycles = 2 * cells
            total_write_cycles = 2 * cells

        total_cycle = total_read_cycles + total_write_cycles

        total_elements = self.dim_A * self.dim_B * self.dim_C * self.dim_D
        computation = (total_read_cells + total_write_cells) * get_elements_num_in_cell(self.x_in_torch_dtype)  # treat each element movement(read + write) as a "computation" for simplicity

        compute_event = ComputeEvent(
            name="Transposition",
            parent=core.transpose.full_name,
            compute_type=EventType.TRANSPOSE,
            computation=computation,
            theoretical_computation=total_elements * 2, # each element is read and written once
            max_consume_rate=computation / total_cycle,
            energy=0, # consider the data movement doesn't consume energy in this event
            precision=self.transpose_type,
        )
        events.append(compute_event)

        memory_width = core.config["core"]["memory_width"]
        L0_read_volume = total_read_cells * memory_width
        L0_write_volume = total_write_cells * memory_width

        L0_read_event = MemoryEvent(
            name="Transposition_read_L0",
            parent=core.memory[0].full_name,
            memory_type=EventType.READ,
            volume=L0_read_volume,
            bounded_events=[],
            energy=np.ceil(L0_read_volume / memory_width) * core.config["core"]["L0_memory_read_energy"],
            max_bandwidth=L0_read_volume / total_cycle,
            hierarchy=0,
        )
        events.append(L0_read_event)

        L0_write_event = MemoryEvent(
            name="Transposition_write_L0",
            parent=core.memory[0].full_name,
            memory_type=EventType.WRITE,
            volume=L0_write_volume,
            bounded_events=[],
            energy=np.ceil(L0_write_volume / memory_width) * core.config["core"]["L0_memory_write_energy"],
            max_bandwidth=L0_write_volume / total_cycle,
            hierarchy=0,
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