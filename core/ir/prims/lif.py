from core.ir.prims.prim import Primitive
from basics.addr_blocks_in_mem import AddrBlocksInMem
from basics.data_block import DataBlock
from basics.utils import num_to_hex
from core.simulator.emulator.core import Core as EmuCore
from core.simulator.analyser.core import Core as AnalyCore
from core.simulator.analyser.event import EventType, ComputeEvent, MemoryEvent
from myhdl import bin, intbv
import torch
import math
from core.utils import get_byte_num

class PrimLif(Primitive):
    def __init__(self, deps, rst_mode, output_mode, Vmp_update_addr, 
                 Tw_en, Vin_addr, Vmp_addr, Vtheta_addr, Sout_addr, para_addr,
                 Vmp_rest, Vmp_rst, Vthr0, Vtheta_rst, Tw_len, Tw_cnt, 
                 Vtheta_incre, Vmp_low, Vthr_low, Vmp_att, Vmp, Vtheta, 
                 A_leaky, B_leaky, A_theta, B_theta, input_num, input_len_cell):
        
        super().__init__()
        self.name = "Lif"

        self.deps = deps
        self.rst_mode = rst_mode
        self.output_mode = output_mode
        self.Tw_en = Tw_en
        self.Vin_addr = Vin_addr
        self.Vmp_addr = Vmp_addr
        self.Vtheta_addr = Vtheta_addr
        self.Sout_addr = Sout_addr
        self.input_num = input_num
        self.input_len_cell = input_len_cell
        self.para_addr = para_addr
        self.Vmp_update_addr = Vmp_update_addr

        self.Vmp_rest = torch.tensor(Vmp_rest, dtype=torch.bfloat16)
        self.Vmp_rst = torch.tensor(Vmp_rst, dtype=torch.bfloat16)
        self.Vthr0 = torch.tensor(Vthr0, dtype=torch.bfloat16)
        self.Vtheta_rst = torch.tensor(Vtheta_rst, dtype=torch.bfloat16)
        self.A_theta = torch.tensor(A_theta, dtype=torch.bfloat16)
        self.B_theta = torch.tensor(B_theta, dtype=torch.bfloat16)
        self.Vtheta_incre = torch.tensor(Vtheta_incre, dtype=torch.bfloat16)
        self.Vmp_low = torch.tensor(Vmp_low, dtype=torch.bfloat16)
        self.Vthr_low = torch.tensor(Vthr_low, dtype=torch.bfloat16)
        self.A_leaky = torch.tensor(A_leaky, dtype=torch.bfloat16)
        self.B_leaky = torch.tensor(B_leaky, dtype=torch.bfloat16)
        self.Vmp_att = torch.tensor(Vmp_att, dtype=torch.bfloat16)
        self.Tw_len = Tw_len
        self.Tw_cnt = Tw_cnt

        self.para = intbv(0, min=0, max=(1<<256))
        self.para[16:0] = int(num_to_hex(self.Vmp_rest, "BF16"), 16)
        self.para[32:16] = int(num_to_hex(self.Vmp_rst, "BF16"), 16)
        self.para[48:32] = int(num_to_hex(self.Vthr0, "BF16"), 16)
        self.para[64:48] = int(num_to_hex(self.Vtheta_rst, "BF16"), 16)
        self.para[80:64] = int(num_to_hex(self.A_theta, "BF16"), 16)
        self.para[96:80] = int(num_to_hex(self.B_theta, "BF16"), 16)
        self.para[112:96] = int(num_to_hex(self.Vtheta_incre, "BF16"), 16)
        self.para[128:112] = int(num_to_hex(self.Vmp_low, "BF16"), 16)
        self.para[144:128] = int(num_to_hex(self.Vthr_low, "BF16"), 16)
        self.para[160:144] = int(num_to_hex(self.A_leaky, "BF16"), 16)
        self.para[176:160] = int(num_to_hex(self.B_leaky, "BF16"), 16)
        self.para[192:176] = int(num_to_hex(self.Vmp_att, "BF16"), 16)
        self.para[208:192] = self.Tw_len
        self.para[224:208] = self.Tw_cnt
        self.data_blocks["para"] = DataBlock(data=self.para, length=1, zero=0, addressing="32B")
        
        # 生成PIC
        self.setPIC()
        
        
    def setPIC(self):
        self.PIC = intbv(0, min=0, max=(1<<256))
        self.PIC[4:0] = 0x3
        self.PIC[8:4] = 0x6
        self.PIC[16:8] = self.deps
        self.PIC[18:16] = self.rst_mode
        self.PIC[21:18] = self.output_mode
        self.PIC[22:21] = self.Tw_en
        self.PIC[47:32] = self.Vin_addr
        self.PIC[63:48] = self.Vmp_addr
        self.PIC[79:64] = self.Vtheta_addr
        self.PIC[95:80] = self.Sout_addr
        self.PIC[111:96] = self.para_addr
        self.PIC[127:112] = self.Vmp_update_addr
        self.PIC[144:128] = self.input_num
        self.PIC[160:144] = self.input_len_cell
        
    def execute(self, core: EmuCore):
        # Get the cells of Vin
        # Unified data length
        data_length = core.memory.getTensorLen((self.input_num, self.input_len_cell * 16), torch.bfloat16, (0, 1))
        # Permute Vin tensor to match the 2D memory shape
        Vin_tensor = core.memory[self.Vin_addr : self.Vin_addr + data_length].view(torch.bfloat16).reshape((self.input_num, -1))[:, : self.input_len_cell * 16].permute(0, 1)
        # Uinfied data shape
        data_shape = Vin_tensor.shape

        # Get Vmp and Vtheta
        if self.Tw_en and not self.Tw_cnt:
            # Reset Vmp_tensor
            Vmp_tensor = torch.full(
                size=data_shape,
                fill_value=self.Vmp_rest,
                dtype=Vin_tensor.dtype
            )
            # Reset Vtheta_tensor
            Vtheta_tensor = torch.full(
                size=data_shape,
                fill_value=self.Vtheta_rst,
                dtype=Vin_tensor.dtype
            )
        else:
            # Permute Vmp tensor to match the 2D memory shape
            Vmp_tensor = core.memory[self.Vmp_addr:self.Vmp_addr + data_length].view(torch.bfloat16).reshape((self.input_num, -1))[:, : self.input_len_cell * 16].permute(0, 1)
            # Permute Vtheta tensor to match the 2D memory shape
            Vtheta_tensor = core.memory[self.Vtheta_addr:self.Vtheta_addr + data_length].view(torch.bfloat16).reshape((self.input_num, -1))[:, : self.input_len_cell * 16].permute(0, 1)

        # Define output tensor Sout_tensor
        if self.output_mode == 0 or self.output_mode == 1:
            Sout_tensor = torch.full(
                size=data_shape,
                fill_value=0,
                dtype=torch.bfloat16
            )
        elif self.output_mode == 2:
            Sout_tensor = torch.full(
                size=data_shape,
                fill_value=0,
                dtype=torch.bool
            )
        elif self.output_mode == 3:
            Sout_tensor = torch.full(
                size=data_shape,
                fill_value=0,
                dtype=torch.int8
            )
        else:
            raise ValueError(f"不支持的输出模式: {self.output_mode}")
        
        # Update Vmp
        Vmp_update_tensor = self.A_leaky * Vmp_tensor + self.B_leaky * (Vin_tensor)
        Vtheta_update_tensor = self.A_theta * Vtheta_tensor + self.B_theta

        # Compare lower bound
        low_mask = Vmp_update_tensor < self.Vmp_low
        Vmp_update_tensor[low_mask] = self.Vmp_low

        # Emit spike
        spike_mask = Vmp_update_tensor >= Vtheta_update_tensor

        if spike_mask.any():
            # Different output modes
            if self.output_mode == 0:
                Sout_tensor[spike_mask] = Vmp_update_tensor[spike_mask]
            elif self.output_mode == 1:
                Sout_tensor[spike_mask] = Vmp_update_tensor[spike_mask] - self.Vmp_att
            elif self.output_mode == 2:
                Sout_tensor[spike_mask] = True
            elif self.output_mode == 3:
                Sout_tensor[spike_mask] = 1

            # Different reset modes
            if self.rst_mode == 0:
                Vmp_update_tensor[spike_mask] = self.Vmp_rst
            elif self.rst_mode == 1:
                Vmp_update_tensor[spike_mask] = Vmp_update_tensor[spike_mask] - self.Vmp_att
            elif self.rst_mode == 2:
                Vmp_update_tensor[spike_mask] = Vmp_update_tensor[spike_mask] - (Vtheta_update_tensor[spike_mask] + self.Vthr0)
            elif self.rst_mode == 3:
                Vmp_update_tensor[spike_mask] = Vmp_update_tensor[spike_mask]

            # Update Vtheta
            Vtheta_update_tensor[spike_mask] += self.Vtheta_incre
        
        # Write the Sout_tensor to the memory
        core.memory.writeTensor(self.Sout_addr, Sout_tensor)
        # Write the Vmp_update_tensor to the memory
        core.memory.writeTensor(self.Vmp_update_addr, Vmp_update_tensor)
        # Write the Vtheta_update_tensor to the memory
        core.memory.writeTensor(self.Vtheta_addr, Vtheta_update_tensor)
        # Read para and update Tw_cnt
        para = core.memory[self.para_addr]
        if self.Tw_en:
            if para[13] < (self.Tw_len - 1):
                para[13] += 1
            else:
                para[13] = 0
        # Write the para to the memory
        core.memory[self.para_addr] = para

    def generate_events(self, core: AnalyCore):
        events = []
        cycles_per_cell = 24
        # if Tw_en than write para
        total_cycle = self.input_num * self.input_len_cell * cycles_per_cell + 2 + (1 if self.Tw_en else 0)
        # 计算过程：Vmp*A_leaky + B_leaky + Vin compare Vtheta*A_theta + B_theta Vtheta + Vthr0 compare compare Vmp - Vmp_att Vtheta + Vtheta_incre Vmp - Vthr0
        # 总计算量
        add_operation = 5 * self.input_num * self.input_len_cell * core.config["core"]["vec_parallelism"]
        mult_add_operation = 2 * self.input_num * self.input_len_cell * core.config["core"]["vec_parallelism"]
        compare_operation = 3 * self.input_num * self.input_len_cell * core.config["core"]["vec_parallelism"]
        computation =  add_operation + mult_add_operation + compare_operation

        # compute event
        compute_event = ComputeEvent(
            name="Lif_compute",
            parent=core.vector.full_name,
            compute_type=EventType.VECTOR,
            computation=computation,
            theoretical_computation=computation,
            max_consume_rate=computation / total_cycle,
            energy=(add_operation + compare_operation) * core.config["core"]["bf16_PE_add_energy"] + mult_add_operation * core.config["core"]["bf16_PE_energy"],
        )
        events.append(compute_event)

        # read event
        L0_volume = 0
        # read Vin / Vmp / Vtheta
        L0_volume += self.input_num * self.input_len_cell * 16 * get_byte_num(torch.bfloat16) * 3
        L0_read_event = MemoryEvent(
            name="Lif_read_L0",
            parent=core.memory[0].full_name,
            memory_type=EventType.READ,
            volume=L0_volume,
            bounded_events=[],
            energy=math.ceil(L0_volume / core.config["core"]["memory_width"]) * core.config["core"]["L0_memory_read_energy"], 
            max_bandwidth=L0_volume / total_cycle,
            hierarchy=0
        )
        events.append(L0_read_event)

        # write event
        output_volume = 0
        # write Sout / Vmp / Vtheta
        if self.output_mode in [0, 1]:
            output_volume += self.input_num * self.input_len_cell * 16 * get_byte_num(torch.bfloat16)
        elif self.output_mode == 2:
            output_volume += self.input_num * math.ceil(self.input_len_cell * 16 / 256) * 32
        elif self.output_mode == 3:
            output_volume += self.input_num * math.ceil(self.input_len_cell * 16 / 32) * 32
        output_volume += self.input_num * self.input_len_cell * 16 * get_byte_num(torch.bfloat16) * 2
        output_event = MemoryEvent(
            name="Lif_write",
            parent=core.memory[0].full_name,
            memory_type=EventType.WRITE,
            volume=output_volume,
            bounded_events=[],
            energy=(math.ceil(output_volume / core.config["core"]["memory_width"]) * core.config["core"]["L0_memory_write_energy"]),
            max_bandwidth=output_volume / total_cycle,
            hierarchy=0
        )
        events.append(output_event)

        # Events excepts Read/Write/Inout
        event_nms = []
        for event in events:
            event_nms.append(event.full_name)
        for event in events:
            if event.event_type in [EventType.READ, EventType.INOUT, EventType.WRITE]:
                event.bounded_events = event_nms
                event.bounded_events.remove(event.full_name)
        
        return events

