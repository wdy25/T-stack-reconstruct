import torch
import numpy as np
import struct

from copy import deepcopy
from myhdl import bin, intbv

from core.ir.prims.prim import Primitive
from basics.addr_blocks_in_mem import AddrBlocksInMem
from basics.data_block import DataBlock
from core.simulator.emulator.core import Core as EmuCore
from core.simulator.analyser.core import Core as AnalyCore
from core.simulator.analyser.event import EventType, ComputeEvent, MemoryEvent
from core.utils.get_byte_num import get_torch_dtype_from_type_num, get_elements_num_in_cell, get_byte_num

class PrimPooling(Primitive):
    def __init__(self, deps, x_in_addr, bias_value_or_addr, out_addr, 
                 para_addr, batch_size, x_in_h, x_in_w, c_in_32B, 
                 kernel_h, kernel_w, scaler, scaler_mode, max_or_min,
                 y_type, bias_mode, stride_h, stride_w, 
                 x_in: DataBlock=None, bias: DataBlock=None):
        
        super().__init__()
        self.name = "Pooling"  # 定义名称
        
        # 记录参数，即原语中的所有字段，包括para中某些需要额外配置的字段
        # PI_code
        self.deps = deps
        self.x_in_addr = x_in_addr
        self.bias_value_or_addr = bias_value_or_addr
        self.out_addr = out_addr
        self.batch_size = batch_size
        self.x_in_h = x_in_h
        self.x_in_w = x_in_w
        self.c_in_32B = c_in_32B
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.scaler = scaler
        self.scaler_mode = scaler_mode
        # x_in_type = DataType.BF16
        self.y_type = y_type
        self.bias_mode = bias_mode
        self.max_or_min = max_or_min
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.para_addr = para_addr
        # PI_Para
        self.y_out_h = ((self.x_in_h + 1) - (self.kernel_h + 1) + (self.stride_h + 1)) // (self.stride_h + 1)
        self.y_out_w = ((self.x_in_w + 1) - (self.kernel_w + 1) + (self.stride_w + 1)) // (self.stride_w + 1)
        # torch dtype
        self.x_in_torch_dtype = torch.bfloat16  # 输入固定为BF16
        self.y_out_torch_dtype = get_torch_dtype_from_type_num(self.y_type)


        self.x_in_len = int((self.batch_size+1) * (self.x_in_h+1) * (self.x_in_w+1) * (self.c_in_32B+1))
        self.bias_in_len = int((self.c_in_32B+1) if self.bias_mode==2 else 1)
        self.para_len = 1
        self.out_len = int((self.batch_size+1) * self.y_out_h * self.y_out_w * (self.c_in_32B+1))
        
        
        # 保存原语需要提前配置的数据，如果有para则必须在此生成
        # 对于非para的数据，可以有3种情况：
        # 1. 为None，表示不需要提前配置，可能是其他原语的输出作为输入，直接跳过这份数据即可
        # 2. 为[]，表示需要提前配置，需要生成一个DataBlock，长度根据原语参数确定，zero=0表示全随机
        # 3. 为DataBlock，表示直接使用这个DataBlock 
        # if data0 is not None:  # 如果data0为None，表示不需要提前配置，不需要处理
        #     if data0 == []:
        #         self.data_blocks["data0"] = DataBlock(data=None, length=param0, zero=0, addressing="32B")
        #     else:
        #         self.data_blocks["data0"] = deepcopy(data0)

        if x_in is not None:
            if x_in == []:
                self.data_blocks["x_in"] = DataBlock(data=None, length=self.x_in_len, zero=0, addressing="32B")
            else:
                self.data_blocks["x_in"] = deepcopy(x_in)
        
        if bias is not None:
            if bias == []:
                self.data_blocks["bias"] = DataBlock(data=None, length=self.bias_in_len, zero=1, addressing="32B")
            else:
                self.data_blocks["bias"] = deepcopy(bias)
        
        
        # 生成para
        self.y_out_h = int(((self.x_in_h + 1) - (self.kernel_h + 1) + (self.stride_h + 1)) / (self.stride_h + 1))
        self.y_out_w = int(((self.x_in_w + 1) - (self.kernel_w + 1) + (self.stride_w + 1)) / (self.stride_w + 1))
        
        self.Kw_offset_in = int((self.c_in_32B + 1))
        self.Kh_offset_in = int((self.c_in_32B + 1) * ((self.x_in_w + 1) - self.kernel_w))
        self.C_offset_in = int(-self.kernel_w * (self.c_in_32B + 1) - self.kernel_h * (self.x_in_w + 1) * (self.c_in_32B + 1) + 1)
        self.Ow_offset_in = int(self.C_offset_in + self.stride_w * (self.c_in_32B + 1))
        self.Oh_offset_in = int(self.Ow_offset_in - self.y_out_w * (self.stride_w + 1) * (self.c_in_32B + 1) + (self.stride_h + 1) * (self.x_in_w + 1) * (self.c_in_32B + 1))
        self.Bc_offset_in = int(self.Oh_offset_in - self.y_out_h * (self.stride_h + 1) * (self.x_in_w + 1) * (self.c_in_32B + 1) + (self.x_in_h + 1) * (self.x_in_w + 1) * (self.c_in_32B + 1))

        # self.Sli_win_kernel = int((self.kernel_h + 1) * (self.kernel_w + 1))
        # self.Addr_start_offset = int(- (self.pad_left * self.Kw_offset_in + self.pad_up * self.kernel_w * self.Kw_offset_in	+ self.pad_up * self.Kh_offset_in))
        # intbv:遵循Python约定，min是包含在内的，max是不包含在内的。因此，允许的值范围是[min,max-1]。
        self.para = intbv(0, min=0, max=(1<<256))
        # intbv: The slice is exclusive of the MSB and inclusive of the LSB (即：大索引不包含，小索引包含)
        self.para[15:0] = intbv(self.Kw_offset_in, min=-(1<<14), max=(1<<14))[15:0]
        self.para[31:16] = intbv(self.Kh_offset_in, min=-(1<<14), max=(1<<14))[15:0]
        self.para[47:32] = intbv(self.C_offset_in, min=-(1<<14), max=(1<<14))[15:0]
        self.para[63:48] = intbv(self.Ow_offset_in, min=-(1<<14), max=(1<<14))[15:0]
        self.para[79:64] = intbv(self.Oh_offset_in, min=-(1<<14), max=(1<<14))[15:0]
        self.para[95:80] = intbv(self.Bc_offset_in, min=-(1<<14), max=(1<<14))[15:0]
        self.para[112:96] = intbv(self.y_out_h, min=0, max=(1<<16))[16:0]
        self.para[128:112] = intbv(self.y_out_w, min=0, max=(1<<16))[16:0]
        # self.para[144:128] = intbv(self.Sli_win_kernel, min=0, max=(1<<16))[16:0]
        # self.para[160:144] = intbv(self.Addr_start_offset, min=-(1<<15), max=(1<<15))[16:0]

        self.data_blocks["para"] = DataBlock(data=self.para, length=self.para_len, zero=0, addressing="32B")


        # 计算地址 起始地址和数据长度
        self.setAddrBlocks()
        
        # 生成PIC
        self.setPIC()
        
    def setPIC(self):
        self.PIC = intbv(0, min=0, max=(1<<256))
        # intbv: The slice is exclusive of the MSB and inclusive of the LSB (即：大索引不包含，小索引包含)
        # PIC Code: 0x53 for Soma module, max pooling
        self.PIC[8:0] = 0x53
        # dep1-8: instruction dependencies
        self.PIC[16:8] = self.deps
        # x_in_addr: input data address (15 bits)
        self.PIC[31:16] = self.x_in_addr
        # bias_value_or_addr: depends on bias_mode
        #   bias_mode==0: write 0
        #   bias_mode==1: write scalar in BF16
        #   bias_mode==2: write address directly
        if int(self.bias_mode) == 0:
            self.PIC[48:32] = 0
        elif int(self.bias_mode) == 1:
            self.PIC[48:32] = self.scalar_to_bf16(self.bias_value_or_addr)
        else:
            self.PIC[47:32] = self.bias_value_or_addr
        # out_addr: output data address (15 bits)
        self.PIC[63:48] = self.out_addr
        # batch_size: input vector length (16 bits, 0 means 1)
        self.PIC[96:80] = self.batch_size
        # x_in_h: input vector length (16 bits, 0 means 1)
        self.PIC[112:96] = self.x_in_h
        # x_in_w: input vector length (16 bits, 0 means 1)
        self.PIC[128:112] = self.x_in_w
        # c_in_32B: input vector, number of rows (16 bits, 0 means 1)
        self.PIC[144:128] = self.c_in_32B
        # kernel_h: kernel size (8 bits, 0 means 1)
        self.PIC[152:144] = self.kernel_h
        # kernel_w: kernel size (8 bits, 0 means 1)
        self.PIC[160:152] = self.kernel_w
        # scaler: scaling factor (7 bits, signed integer) it means output_final = output * 2^scaler
        self.PIC[167:160] = intbv(self.scaler, min=-(1<<6), max=(1<<6))[7:0]
        # scaler_mode: scaling enable (1 bit, 0: disable, 1: enable)
        self.PIC[167] = self.scaler_mode
        # max_or_min: pooling type (1 bit, 0: Max Pooling, 1: Min Pooling)
        self.PIC[168] = self.max_or_min
        # y_type: output type (1 bit, 0: INT8, 1: BF16)
        self.PIC[171] = self.y_type
        # bias_mode: bias mode (2 bits, 0: no bias, 1: scalar bias, 2: bias vector from memory)
        self.PIC[174:172] = self.bias_mode
        # stride_h: h direction stride (8 bits, 0 means 1)
        self.PIC[200:192] = self.stride_h
        # stride_w: w direction stride (8 bits, 0 means 1)
        self.PIC[208:200] = self.stride_w
        # para_addr: PL_para address (15 bits)
        self.PIC[255:240] = self.para_addr
        
        
        
    def setAddrBlocks(self):
        # 除了各种输入数据外，还需要记录结果数据的地址、Para的地址等各种运行中需要的数据地址核长度
        # 此函数不应该直接使用self.data_blocks中的数据长度，因为这里计算的长度需要和self.data_blocks中的数据长度进行对比以确保传入的数据长度是正确的
        
        # self.data_addr_list["data_name"] = AddrBlocksInMem({addr: length}, addressing)
        
        self.data_addr_list["x_in"] = AddrBlocksInMem({self.x_in_addr: self.x_in_len}, "32B")
        if self.bias_mode == 2:
            self.data_addr_list["bias"] = AddrBlocksInMem({self.bias_value_or_addr: self.bias_in_len}, "32B")
        self.data_addr_list["para"] = AddrBlocksInMem({self.para_addr: self.para_len}, "32B")
        self.data_addr_list["out"] = AddrBlocksInMem({self.out_addr: self.out_len}, "32B")

    def scalar_to_bf16(self, value):
        """Convert a scalar to BF16 16-bit integer representation."""
        if isinstance(value, int):
            value = float(value)
        float32_bytes = struct.pack('>f', float(value))
        bf16_int = struct.unpack('>H', float32_bytes[0:2])[0]
        return bf16_int

    def handwritten_maxpooling(self, input_tensor, init_value, batch, out_h, out_w, in_h, in_w, channels, kernel_h, kernel_w, stride_h, stride_w):
        output = torch.zeros((batch, out_h, out_w, channels), dtype=torch.bfloat16)

        for b in range(batch):
            for oh in range(out_h):
                for ow in range(out_w):
                    # Calculate starting position of the input window
                    h_start = oh * stride_h
                    w_start = ow * stride_w

                    # Use bias as the initial value
                    if self.bias_mode == 0:
                        # No bias, use the cell of the current pooling window at position(0, 0) as the initial value; For simplicity, we use the all cells of the current pooling window at position(0, 0).
                        pooled_value = input_tensor[b, h_start, w_start, :]
                    else:
                        pooled_value = init_value.clone()

                    # Traverse kernel window
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            if (self.bias_mode == 0 and (kh == 0 and kw == 0)):
                                # the first cell of the current pooling window has been used as the initial value when bias_mode == 0, skip it in the pooling calculation to avoid being used twice
                                continue

                            ih = h_start + kh
                            iw = w_start + kw

                            if 0 <= ih < in_h and 0 <= iw < in_w:
                                window_value = input_tensor[b, ih, iw, :]
                                if self.max_or_min == 0:  # Max Pooling
                                    pooled_value = torch.maximum(pooled_value, window_value)
                                else:  # Min Pooling
                                    pooled_value = torch.minimum(pooled_value, window_value)

                    output[b, oh, ow, :] = pooled_value

        return output
    
    def execute(self, core: EmuCore):
        # Input data shape: (batch_size+1, x_in_h+1, x_in_w+1, (c_in_32B+1) * 16) for BF16
        # Input is fixed as BF16 type, each 32B unit contains 16 BF16 elements
        batch = self.batch_size + 1
        in_h = self.x_in_h + 1
        in_w = self.x_in_w + 1
        channels = (self.c_in_32B + 1) * 16  # BF16: 16 elements per 32B
        kernel_h = self.kernel_h + 1
        kernel_w = self.kernel_w + 1
        stride_h = self.stride_h + 1
        stride_w = self.stride_w + 1
        
        # Read input tensor
        input_tensor_size = (batch, in_h, in_w, channels)
        input_tensor = core.memory.readTensor(
            self.x_in_addr, self.x_in_len, input_tensor_size, torch.bfloat16, (0, 1, 2, 3)
        )  # shape: (batch, in_h, in_w, channels)
        
        # Prepare bias as initial value for pooling
        if self.bias_mode == 0:
            # No bias, use the first cell of the current pooling window as the initial value
            pass
        elif self.bias_mode == 1:
            # bias is a scalar value (stored in bias_value_or_addr)
            bias_scalar = self.bias_value_or_addr
            init_value = torch.full((channels,), bias_scalar, dtype=torch.bfloat16)
        else:  # bias_mode == 2
            # bias read from memory, shape: (c_in_32B+1) * 16 -> (channels,)
            bias_tensor_size = ((self.c_in_32B + 1) * 16,)
            bias_tensor = core.memory.readTensor(
                self.bias_value_or_addr, self.c_in_32B + 1, bias_tensor_size, torch.bfloat16
            )
            init_value = bias_tensor  # shape: (channels,)
        
        # Perform pooling
        out_h = self.y_out_h
        out_w = self.y_out_w
        output = self.handwritten_maxpooling(
            input_tensor=input_tensor,
            init_value=init_value if self.bias_mode != 0 else None,
            batch=batch,
            out_h=out_h,
            out_w=out_w,
            in_h=in_h,
            in_w=in_w,
            channels=channels,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            stride_h=stride_h,
            stride_w=stride_w,
        )
        
        # Apply scaler
        if self.scaler_mode == 1:
            scale_factor = 2.0 ** self.scaler
            output = output * scale_factor
        
        # Convert output type
        if self.y_out_torch_dtype == torch.int8:  # INT8
            output = output.clamp(-128, 127).to(torch.int8)
        elif self.y_out_torch_dtype == torch.bfloat16:  # BF16
            output = output.to(torch.bfloat16)
        
        # Write the output tensor to memory
        core.memory.writeTensor(self.out_addr, output, (0, 1, 2, 3))
    
    def generate_events(self, core: AnalyCore):
        events = []
        window_size = (self.kernel_h + 1) * (self.kernel_w + 1)
        # computation
        if self.bias_mode == 0:
            computation_per_cell_window = (window_size - 1) * get_elements_num_in_cell(self.x_in_torch_dtype)  # the number of comparison for each pooling window based on cell ; -1 is because the first cell is used as initial value when bias_mode == 0
        else:
            computation_per_cell_window = window_size * get_elements_num_in_cell(self.x_in_torch_dtype)
        
        total_cell_windows = (self.batch_size + 1) * (self.y_out_h) * (self.y_out_w) * (self.c_in_32B + 1) # the total number of pooling windows based on cell
        computation = total_cell_windows * computation_per_cell_window
        
        # cycles
        if self.bias_mode == 0:
            cycles_per_cell_window_until_compare = 2 + (window_size - 1) * 3 # read_first + (n - 1) * (read + compare) = 2 + (n - 1) * (2 + 1), where n is the size of pooling window.
        elif self.bias_mode == 1:
            cycles_per_cell_window_until_compare = window_size * 3 # n * (read + compare) = n * (2 + 1), where n is the size of pooling window.
        elif self.bias_mode == 2:
            cycles_per_cell_window_until_compare = 2 + window_size * 3 # read_bias_A + n * (read + compare) = 2 + n * (2 + 1), where n is the size of pooling window.

        elements_in_channel = (self.c_in_32B + 1) * get_elements_num_in_cell(self.x_in_torch_dtype)
        total_output_cells = (self.batch_size + 1) * self.y_out_h * self.y_out_w * np.ceil(elements_in_channel / get_elements_num_in_cell(self.y_out_torch_dtype))

        convert_mode = 1 if self.y_out_torch_dtype == torch.int8 else 0
        total_cycle = total_cell_windows * cycles_per_cell_window_until_compare + total_output_cells * (self.scaler_mode*1 + convert_mode*1 + 1)  # enabled scaling needs 1 cycle + enabled type conversion needs 1 cycle + write, the three parts are need per output cell.
        
        compute_event = ComputeEvent(
            name="MaxPooling",
            parent=core.vector.full_name,
            compute_type=EventType.VECTOR,
            computation=computation,
            theoretical_computation=computation, # The theoretical computation equals the actual computation because it is aligned by cell.
            max_consume_rate=computation / total_cycle,
            energy=computation * core.config["core"]["bf16_PE_add_energy"], # use BF16 PE add energy as BF16 comparison energy
        )
        events.append(compute_event)
        
        # read
        input_cells_per_cell_window = window_size
        total_read_input_cells = total_cell_windows * input_cells_per_cell_window
        total_read_bias_cells = total_cell_windows  if self.bias_mode == 2 else 0
        L0_read_volume = (total_read_input_cells + total_read_bias_cells) * core.config["core"]["memory_width"] # the volume of data read from L0, in bytes
            
        L0_read_event = MemoryEvent(
            name="MaxPooling_read_L0",
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
        L0_write_volume = total_output_cells * core.config["core"]["memory_width"] # the volume of data written to L0 memory, in bytes
        
        L0_write_event = MemoryEvent(
            name="MaxPooling_write_L0",
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
