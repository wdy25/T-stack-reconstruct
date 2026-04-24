from core.ir.prims.prim import Primitive
from core.simulator.emulator.core import Core as EmuCore
from core.simulator.analyser.core import Core as AnalyCore
from core.simulator.analyser.event import EventType, ComputeEvent, MemoryEvent
from basics.addr_blocks_in_mem import AddrBlocksInMem
from basics.data_block import DataBlock

from copy import deepcopy
from myhdl import bin, intbv
import numpy as np
import torch

from core.ir.utils.handwritten_dendrite_operator import handwritten_matmul
from core.utils.get_byte_num import get_elements_num_in_cell, get_byte_num

def to_signed_bits(val, width=16):
    if not -(1 << (width-1)) <= val < (1 << (width-1)):
        raise ValueError("超出范围: %d位有符号数" % width)
    return val & ((1 << width) - 1)  # 得到补码

class PrimMatrixMultiplication(Primitive):
    """Matrix Multiplication Prim.

    Ports:
        inputs:
            - 0: input tensor 3D (Dim_A, Batch_size, C_in).
            - 1: weight tensor 3D (Dim_A, C_in, C_out).
            - 2: bias tensor 1D (C_out,).
        outputs:
            - 0: output tensor 3D (Dim_A, Batch_size, C_out).

    Attributes:
        matmul_type (str): Specifies the weight data type ('INT8', 'BF16', 'SPIKE').
        isSNN (bool): Controls the input 'x' data type. If True, 'x' is SPIKE type; otherwise, it matches 'matmul_type'.
        deps (int): Dependency bits for scheduling.
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        dim_A (int): Number of slices in the first dimension.
        x_in_addr (int): Memory address for input 'x'.
        w_in_addr (int): Memory address for weight 'w'.
        b_in_addr (int): Memory address for bias 'b'.
        y_out_addr (int): Memory address for output 'y'.
        batch_size (int): Batch size.
        param_addr_1 (int): Memory address for extended parameter fields.

    Supported Data Type Combinations:
        The Prim supports the following combinations of input, weight, and bias data types:
            - input: INT8, weight: INT8, bias: INT32, output: INT32->BF16
            - input: SPIKE, weight: INT8, bias: INT32, output: INT32->BF16
            - input: BF16, weight: BF16, bias: BF16, output: BF16
            - input: SPIKE, weight: BF16, bias: BF16, output: BF16
            - input: SPIKE, weight: SPIKE, bias: INT32, output: INT32->BF16
        Output type rules:
            - The final output data type exposed to the outside is always BF16:
                - When the internal output is INT32, it should be converted to BF16.
                - When the internal output is BF16, no conversion is needed.
        
    """
    def __init__(self,
        matmul_type,
        isSNN,
        deps,
        c_in,
        c_out,
        dim_A,
        x_in_addr,
        w_in_addr,
        b_in_addr,
        y_out_addr,
        batch_size,
        param_addr_1,
        # b_in_data,
        # x_in_data,
        # w_in_data
        ):
        super().__init__()
        self.name = "MatrixMultiplication"
        self.matmul_type = matmul_type
        self.isSNN = isSNN
        self.dim_A = dim_A
        if self.matmul_type == 'INT8':
            self.w_dw = 8
            if self.isSNN:
                self.x_dw = 1
                self.x_type = torch.bool
            else:
                self.x_dw = 8
                self.x_type = torch.int8
            self.w_type = torch.int8
            self.b_dw = 32
            self.b_type = torch.int32
            self.y_dw = 16
            self.y_type = torch.bfloat16
            self.xreg_depth = 16
            self.wreg_width = 256 / self.w_dw
            self.xreg_width = 256 / self.x_dw
        elif self.matmul_type == 'BF16':
            self.w_dw = 16
            if self.isSNN:
                self.x_dw = 1
                self.x_type = torch.bool
            else:
                self.x_dw = 16
                self.x_type = torch.bfloat16
            self.w_type = torch.bfloat16
            self.b_dw = 16
            self.b_type = torch.bfloat16
            self.y_dw = 16
            self.y_type = torch.bfloat16
            self.xreg_depth = 8
            self.wreg_width = 256 / self.w_dw
            self.xreg_width = 256 / self.x_dw
        elif self.matmul_type == 'SPIKE': # Use INT8 MAC
            self.isSNN = True
            self.w_dw = 1
            self.x_dw = 1
            self.x_type = torch.bool
            self.w_type = torch.bool
            self.b_dw = 32
            self.b_type = torch.int32
            self.y_dw = 16
            self.y_type = torch.bfloat16
            self.xreg_depth = 16
            self.wreg_width = 256 / (self.w_dw * 8)
            self.xreg_width = 256 / self.x_dw

        # Internal accumulation type depends on bias type; final outside-visible output is BF16.
        self.y_acc_type = torch.int32 if self.b_type == torch.int32 else torch.bfloat16
        self.y_final_type = torch.bfloat16
        self.deps = deps
        self.c_in = c_in
        self.c_out = c_out
        self.x_in_addr = x_in_addr
        self.w_in_addr = w_in_addr
        self.b_in_addr = b_in_addr
        self.y_out_addr = y_out_addr
        self.batch_size = batch_size
        self.param_addr_1 = param_addr_1

        self.cin_16 = self.c_in if (self.c_in % 16 == 0) else (self.c_in // 16 + 1) * 16
        self.cin_32 = self.c_in if (self.c_in % 32 == 0) else (self.c_in // 32 + 1) * 32
        self.cin_256 = self.c_in if (self.c_in % 256 == 0) else (self.c_in // 256 + 1) * 256
        self.cin_head = self.c_in // self.xreg_width - 1 if (self.c_in % self.xreg_width == 0) else self.c_in // self.xreg_width

        if (self.isSNN == 0):
            if matmul_type == 'INT8':
                self.x_addr_gap_1 = self.cin_32 * self.x_dw / 256
                self.x_addr_gap_2 = -0 + self.xreg_width * self.x_dw / 256
                self.x_addr_gap_3 = -0 - (self.cin_32 / self.xreg_width - 1) * self.xreg_width * self.x_dw / 256
                self.x_addr_gap_4 = -0 - (self.cin_32 / self.xreg_width - 1) * self.xreg_width * self.x_dw / 256 + self.xreg_depth * self.cin_32 * self.x_dw / 256
            elif matmul_type == 'BF16':
                self.x_addr_gap_1 = self.cin_16 * self.x_dw / 256
                self.x_addr_gap_2 = -0 + self.xreg_width * self.x_dw / 256
                self.x_addr_gap_3 = -0 - (self.cin_16 / self.xreg_width - 1) * self.xreg_width * self.x_dw / 256
                self.x_addr_gap_4 = -0 - (self.cin_16 / self.xreg_width - 1) * self.xreg_width * self.x_dw / 256 + self.xreg_depth * self.cin_16 * self.x_dw / 256
        else:
            self.x_addr_gap_1 = self.cin_256 * self.x_dw / 256
            self.x_addr_gap_2 = -0 + self.xreg_width * self.x_dw / 256
            self.x_addr_gap_3 = -0 - (self.cin_256 / self.xreg_width - 1) * self.xreg_width * self.x_dw / 256
            self.x_addr_gap_4 = -0 - (self.cin_256 / self.xreg_width - 1) * self.xreg_width * self.x_dw / 256 + self.xreg_depth * self.cin_256 * self.x_dw / 256

        if self.matmul_type != 'SPIKE':
            self.w_addr_gap_1 = self.c_out * self.w_dw / 256
            self.w_addr_gap_2 = -0 + self.xreg_width * self.c_out * self.w_dw / 256
            self.w_addr_gap_3 = -0 - self.cin_head * self.xreg_width * self.c_out * self.w_dw / 256 + self.wreg_width * self.w_dw / 256
            self.w_addr_gap_4 = -0 - self.cin_head * self.xreg_width * self.c_out * self.w_dw / 256 - (self.c_out / self.wreg_width - 1) * self.wreg_width * self.w_dw / 256
            self.w_addr_gap_5 = 0
        else:
            self.cout_256 = self.c_out if (self.c_out % 256 == 0) else (self.c_out // 256 + 1) * 256
            self.w_addr_gap_1 = self.cout_256 * self.w_dw / 256
            self.w_addr_gap_2 = -0 + self.xreg_width * self.cout_256 * self.w_dw / 256
            self.w_addr_gap_3 = -0 - self.cin_head * self.xreg_width * self.cout_256 * self.w_dw / 256 + self.wreg_width * 8 * self.w_dw / 256
            self.w_addr_gap_4 = -0 - self.cin_head * self.xreg_width * self.cout_256 * self.w_dw / 256 - (self.cout_256 / (self.wreg_width * 8) - 1) * self.wreg_width * 8 * self.w_dw / 256
            self.w_addr_gap_5 = -0 - self.cin_head * self.xreg_width * self.cout_256 * self.w_dw / 256

        if self.matmul_type == 'INT8':
            self.y_addr_gap_0 = 1
        elif self.matmul_type == 'BF16':
            self.y_addr_gap_0 = 0
        elif self.matmul_type == 'SPIKE':
            self.y_addr_gap_0 = 1
        # self.y_addr_gap_1 = -self.y_addr_gap_0 + self.c_out * self.y_dw / 256
        self.y_addr_gap_1 = self.c_out * self.y_dw / 256
        # self.y_addr_gap_2 = -self.y_addr_gap_0 - (self.reg_depth - 1) * self.c_out * self.y_dw / 256 + self.wreg_width * self.y_dw / 256
        self.y_addr_gap_2 = -self.y_addr_gap_0 - 0 + self.wreg_width * self.y_dw / 256
        self.y_addr_gap_3 = -self.y_addr_gap_0 - 0 - (self.c_out / self.wreg_width - 1) * self.wreg_width * self.y_dw / 256 + self.xreg_depth * self.c_out * self.y_dw / 256

        self.b_addr_gap = self.wreg_width * self.b_dw / 256

        self.param1 = intbv(0)[256:]
        self.param1[16:0] = to_signed_bits(int(self.x_addr_gap_1),16)
        self.param1[32:16] = to_signed_bits(int(self.x_addr_gap_2),16)
        self.param1[48:32] = to_signed_bits(int(self.x_addr_gap_3),16)
        self.param1[64:48] = to_signed_bits(int(self.x_addr_gap_4),16)
        self.param1[80:64] = to_signed_bits(int(self.w_addr_gap_1),16)
        self.param1[96:80] = to_signed_bits(int(self.w_addr_gap_2),16)
        self.param1[112:96] = to_signed_bits(int(self.w_addr_gap_3),16)
        self.param1[128:112] = to_signed_bits(int(self.w_addr_gap_4),16)
        self.param1[144:128] = to_signed_bits(int(self.y_addr_gap_1),16)
        self.param1[160:144] = to_signed_bits(int(self.y_addr_gap_2),16)
        self.param1[176:160] = to_signed_bits(int(self.y_addr_gap_3),16)
        self.param1[192:176] = to_signed_bits(int(self.b_addr_gap),16)
        self.param1[208:192] = to_signed_bits(int(self.w_addr_gap_5),16)
        self.data_blocks["param1"] = DataBlock(data=self.param1, length=1, zero=0, addressing="32B")
        
        self.setPIC()

        self.setAddrBlocks()
        
        
    def setPIC(self):
        self.PIC = intbv(0, min=0, max=(1<<256))
        
        self.PIC[4:0] = 0x1
        if self.matmul_type == 'INT8' and self.isSNN == 0:
            self.PIC[8:4] = 0x4
        elif self.matmul_type == 'BF16' and self.isSNN == 0:
            self.PIC[8:4] = 0x5
        elif self.matmul_type == 'INT8' and self.isSNN == 1:
            self.PIC[8:4] = 0xa
        elif self.matmul_type == 'BF16' and self.isSNN == 1:
            self.PIC[8:4] = 0xb
        elif self.matmul_type == 'SPIKE':
            self.PIC[8:4] = 0xc
        
        self.PIC[16:8] = self.deps
        self.PIC[32:16] = self.x_in_addr
        self.PIC[47:32] = self.w_in_addr
        self.PIC[62:47] = self.b_in_addr
        self.PIC[77:62] = self.y_out_addr
        self.PIC[92:77] = self.param_addr_1
        self.PIC[144:138] = self.c_out / self.wreg_width - 1
        self.PIC[172:160] = self.dim_A - 1
        if (self.batch_size % self.xreg_depth == 0):
            self.PIC[182:172] = self.batch_size // self.xreg_depth - 1
        else:
            self.PIC[182:172] = self.batch_size // self.xreg_depth
        self.PIC[218:213] = self.batch_size % self.xreg_depth
        if (self.c_in % self.xreg_width == 0):
            self.PIC[138:132] = self.c_in // self.xreg_width - 1
        else:
            self.PIC[138:132] = self.c_in // self.xreg_width
        self.PIC[226:218] = self.c_in % self.xreg_width
    
    def setAddrBlocks(self):
        # self.data_addr_list["b_in_data"] = AddrBlocksInMem({self.b_in_addr:self.send_cell_num_1}, "32B")
        # self.data_addr_list["x_in_data"] = AddrBlocksInMem({self.x_in_addr:self.send_cell_num_2}, "32B")
        # self.data_addr_list["w_in_data"] = AddrBlocksInMem({self.w_in_addr:self.send_cell_num_3}, "32B")
        # self.data_addr_list["y_out_data"] = AddrBlocksInMem({self.y_out_addr:self.send_cell_num_4}, "32B")
        # self.data_addr_list["param1"] = AddrBlocksInMem({self.param_addr_1: 1}, "32B")
        pass

        
    def execute(self, core: EmuCore):
        # Read input/weight/bias tensors from memory.
        # NOTE: use Memory.readTensor to correctly decode SPIKE (torch.bool) which is bit-packed in memory.
        x_shape = (self.dim_A, self.batch_size, self.c_in)
        w_shape = (self.dim_A, self.c_in, self.c_out)
        b_shape = (self.c_out,)

        x_len = core.memory.getTensorLen(x_shape, self.x_type, (0, 1, 2))
        w_len = core.memory.getTensorLen(w_shape, self.w_type, (0, 1, 2))
        b_len = core.memory.getTensorLen(b_shape, self.b_type, (0,))

        x_tensor = core.memory.readTensor(self.x_in_addr, x_len, x_shape, self.x_type, order=(0, 1, 2))
        w_tensor = core.memory.readTensor(self.w_in_addr, w_len, w_shape, self.w_type, order=(0, 1, 2))
        b_tensor = core.memory.readTensor(self.b_in_addr, b_len, b_shape, self.b_type, order=(0,))

        output_tensor = handwritten_matmul(x_tensor, w_tensor, b_tensor).to(torch.bfloat16)

        # Write output tensor back to memory.
        core.memory.writeTensor(self.y_out_addr, output_tensor, (0, 1, 2))
    
    
    def generate_events(self, core: AnalyCore):
        events = []

        # Use INT8 PE config for INT8/SPIKE; BF16 uses BF16 PE config.
        if self.matmul_type == 'BF16' and self.isSNN == False:
            array_h = int(core.config["core"]["bf16_PE_array_height"])
            array_w = int(core.config["core"]["bf16_PE_array_width"])
            buffer_depth = int(core.config["core"]["bf16_PE_buffer_depth"])
            pe_energy = core.config["core"]["bf16_PE_energy"]
            pe_add_energy = core.config["core"]["bf16_PE_add_energy"]
            precision = "BF16"
        elif self.matmul_type == 'BF16' and self.isSNN == True:
            array_h = int(core.config["core"]["bf16_PE_array_height"])
            array_w = int(core.config["core"]["bf16_PE_array_width"])
            buffer_depth = 256
            pe_energy = core.config["core"]["bf16_PE_energy"]
            pe_add_energy = core.config["core"]["bf16_PE_add_energy"]
            precision = "BF16"
        elif self.matmul_type == 'INT8' and self.isSNN == False:
            array_h = int(core.config["core"]["int8_PE_array_height"])
            array_w = int(core.config["core"]["int8_PE_array_width"])
            buffer_depth = int(core.config["core"]["int8_PE_buffer_depth"])
            pe_energy = core.config["core"]["int8_PE_energy"]
            pe_add_energy = core.config["core"]["int8_PE_add_energy"]
            precision = "INT8"
        else: # x: spike w: int8 or x: spike w: spike
            array_h = int(core.config["core"]["int8_PE_array_height"])
            array_w = int(core.config["core"]["int8_PE_array_width"])
            buffer_depth = 256
            pe_energy = core.config["core"]["int8_PE_energy"]
            pe_add_energy = core.config["core"]["int8_PE_add_energy"]
            precision = "INT8"

        L_dimA = self.dim_A
        L_batch = np.ceil(self.batch_size / array_h)
        L_cout = np.ceil(self.c_out / array_w)
        L_cin = np.floor(self.c_in / buffer_depth)
        cin_res = self.c_in % buffer_depth # cin_res == 0 for INT8xINT8 and BF16xBF16, because their cin is aligned to buffer_depth(32/16). For cases with spike input, cin can be arbitrary, so cin_res can be > 0.

        mac_size = array_h * array_w

        # One inner loop compute cycles equals PE buffer depth.
        inner_compute_cycle = buffer_depth
        # Cells read per inner loop.
        inner_input_cells = array_h
        input_res_cells_per_loop = array_h if cin_res > 0 else 0
        inner_weight_cells = buffer_depth
        # Bias/output per output tile.
        bias_elements_per_loop = array_w
        output_elements_per_loop = array_h * array_w
        convert_elements_per_loop = output_elements_per_loop if self.b_type == torch.int32 else 0

        memory_width = core.config["core"]["memory_width"]

        # Read cycles: read takes 1 cycles because of pipeline, write takes 1 cycle.
        inner_input_cycle = inner_input_cells + 1 # one read needs 2 cycles, but there is overlap between different reads. So the cycles will be number of cells + 1.
        input_res_cycle_per_loop = input_res_cells_per_loop + 1 if input_res_cells_per_loop > 0 else 0

        bias_cycle_per_loop = np.ceil(bias_elements_per_loop / get_elements_num_in_cell(self.b_type)) * 2
        convert_cycle_per_loop = np.ceil(convert_elements_per_loop / get_elements_num_in_cell(self.y_final_type))
        write_cycle_per_loop = np.ceil(output_elements_per_loop / get_elements_num_in_cell(self.y_final_type))
        convert_write_cycle_per_loop = write_cycle_per_loop + 1 if self.b_type == torch.int32 else write_cycle_per_loop # There is overlap between convert and write. So if conversion is needed, the cycles will be write cycles + 1, otherwise just write cycles.

        # Overlap input read and (weight read + computation); MAC finishes 1 cycle after the last weight read.
        inner_cycle = max(inner_input_cycle , inner_compute_cycle + 2) # read the first weight cell before starting computation
        last_overlap_read_mac_cycle_per_loop = max(input_res_cycle_per_loop, inner_compute_cycle + 2)
        res_compute_cycle_per_loop = cin_res

        total_cycle = (
            L_dimA * L_batch * L_cout * (bias_cycle_per_loop + inner_input_cycle + (L_cin - 1) * inner_cycle + last_overlap_read_mac_cycle_per_loop + (res_compute_cycle_per_loop + 2))
            + L_dimA * L_batch * L_cout * convert_write_cycle_per_loop
        )

        if not self.isSNN:
            total_mac_computation = L_dimA * L_batch * L_cout * self.c_in * mac_size * 2 # mul + add
        else:
            total_mac_computation = L_dimA * L_batch * L_cout * self.c_in * mac_size # add only
        total_convert_computation = L_dimA * L_batch * L_cout * convert_elements_per_loop # BF16 conversion when bias is INT32
        total_computation = total_mac_computation + total_convert_computation

        # Energy model: for spike/boolean inputs (and/or boolean weights), treat as add-only.
        mac_energy = total_mac_computation * (pe_add_energy if self.isSNN else pe_energy/2)
        convert_energy = total_convert_computation * pe_add_energy # Assume conversion energy is similar to add energy
        energy = mac_energy + convert_energy

        theoretical_computation = self.dim_A  * (self.c_in + self.c_in) * (self.batch_size * self.c_out) # mul: cin, add: cin because of having initial bias

        compute_event = ComputeEvent(
            name="MatrixMul",
            parent=core.matrix.full_name,
            compute_type=EventType.MATRIX,
            computation=total_computation,
            theoretical_computation=theoretical_computation,
            max_consume_rate=(total_computation / total_cycle) if total_cycle else 0,
            energy=energy,
            precision=precision,
        )
        events.append(compute_event)

        # Only consider L0 memory
        # read event
        L0_read_volume = 0
        # read input
        L0_read_volume += (input_res_cells_per_loop + inner_input_cells * memory_width * L_cin) * L_cout * L_batch * L_dimA
        # read weight
        L0_read_volume += inner_weight_cells * memory_width * self.c_in * np.ceil(self.c_out / get_elements_num_in_cell(self.w_type)) * L_batch * L_dimA
        # read bias (shared across dim_A)
        L0_read_volume += bias_elements_per_loop * get_byte_num(self.b_type) * L_cout * L_batch * L_dimA

        L0_read_event = MemoryEvent(
            name="MatrixMul_read_L0",
            parent=core.memory[0].full_name,
            memory_type=EventType.READ,
            volume=L0_read_volume,
            bounded_events=[],
            energy=np.ceil(L0_read_volume / memory_width) * core.config["core"]["L0_memory_read_energy"],
            max_bandwidth=(L0_read_volume / total_cycle) if total_cycle else 0,
            hierarchy=0,
        )
        events.append(L0_read_event)

        # write event (final output is BF16)
        L0_write_volume = output_elements_per_loop * get_byte_num(self.y_final_type) * L_cout * L_batch * L_dimA
        output_event = MemoryEvent(
            name="MatrixMul_write_L0",
            parent=core.memory[0].full_name,
            memory_type=EventType.WRITE,
            volume=L0_write_volume,
            bounded_events=[],
            energy=np.ceil(L0_write_volume / memory_width) * core.config["core"]["L0_memory_write_energy"],
            max_bandwidth=(L0_write_volume / total_cycle) if total_cycle else 0,
            hierarchy=0,
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
        