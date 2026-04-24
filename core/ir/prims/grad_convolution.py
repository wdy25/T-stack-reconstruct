import torch
import torch.nn.functional as F
from myhdl import bin, intbv
import math, struct
from core.ir.prims.prim import Primitive
from basics.addr_blocks_in_mem import AddrBlocksInMem
from basics.data_block import DataBlock
from core.simulator.emulator.core import Core as EmuCore
from core.simulator.analyser.core import Core as AnalyCore
from core.simulator.analyser.event import EventType, ComputeEvent, MemoryEvent
from core.utils.get_byte_num import get_byte_num

def to_signed_bits(val, width=16):
    if not -(1 << (width-1)) <= val < (1 << (width-1)):
        raise ValueError("超出范围: %d位有符号数" % width)
    return val & ((1 << width) - 1)  # 得到补码

def to_bf16_bits(f: float):
    """
    将一个实数转换为其 bfloat16 编码对应的16位整数值。

    bfloat16 是通过将32位浮点数的符号位、指数位保留，
    并将尾数截断到7位得到的。

    Args:
        f: 输入的实数。

    Returns:
        一个16位的整数，其位模式与输入的 bfloat16 编码一致。
    """
    # 处理特殊值：NaN (非数字)
    if math.isnan(f):
        return 0x7fc1  # bfloat16 NaN 的整数表示

    # 处理特殊值：无穷大
    if math.isinf(f):
        return 0x7f80 if f > 0 else 0xff80

    # 1. 将 Python 的 float (64位) 打包成32位单精度浮点数的字节
    try:
        packed_32 = struct.pack('!f', f)
    except OverflowError:
        # 如果输入的数绝对值太大，无法表示为32位浮点数，则视为无穷大
        return 0x7f80 if f > 0 else 0xff80

    # 2. 将32位字节解包为一个无符号整数，以便进行位操作
    int_32 = struct.unpack('!I', packed_32)[0]

    # 3. 通过右移16位来截断尾数的低16位，得到 bfloat16 的整数表示
    bf16_int = int_32 >> 16

    # 4. 直接返回这个16位整数
    return bf16_int

class PrimGradConvolution(Primitive):
    def __init__(self,
        conv_type,
        isSNN,
        deps,
        x_in_h,
        x_in_w,
        c_in,
        c_out,
        k_h,
        k_w,
        x_in_addr,
        w_in_addr,
        b_in_addr,
        y_out_addr,
        bs,
        dilation_h,
        dilation_w,
        stride_h,
        stride_w,
        padding_top,
        padding_bottom,
        padding_left,
        padding_right,
        padding_value,
        param_addr_1,
        param_addr_2,
        # b_in_data,
        # x_in_data,
        # w_in_data
        ):
        super().__init__()
        self.name = "GradConvolution"  # 定义名称
        self.conv_type = conv_type
        self.isSNN = isSNN
        self.w_dw = 16
        self.x_dw = 16
        self.b_dw = 16
        self.y_dw = 16
        self.xreg_depth = 8
        self.wreg_width = 256 / self.w_dw
        self.xreg_width = 256 / self.x_dw
        self.deps = deps
        self.x_in_h = x_in_h
        self.x_in_w = x_in_w
        self.c_in = c_in
        self.c_out = c_out
        self.k_h = k_h
        self.k_w = k_w
        self.x_in_addr = x_in_addr
        self.w_in_addr = w_in_addr
        self.b_in_addr = b_in_addr
        self.y_out_addr = y_out_addr
        self.bs = bs
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_top = padding_top
        self.padding_bottom = padding_bottom
        self.padding_left = padding_left
        self.padding_right = padding_right
        self.padding_value = padding_value
        self.param_addr_1 = param_addr_1
        self.param_addr_2 = param_addr_2

        self.cin_16 = self.c_in if (self.c_in % 16 == 0) else (self.c_in // 16 + 1) * 16
        self.cin_head = self.c_in // self.xreg_width - 1 if (self.c_in % self.xreg_width == 0) else self.c_in // self.xreg_width

        self.xw_pad = self.x_in_w + self.padding_left + self.padding_right
        self.xh_pad = self.x_in_h + self.padding_top + self.padding_bottom
        self.y_out_w = (self.x_in_w + self.padding_left + self.padding_right - (self.k_w + (self.dilation_w - 1) * (self.k_w - 1))) // self.stride_w + 1
        self.y_out_h = (self.x_in_h + self.padding_top + self.padding_bottom - (self.k_h + (self.dilation_h - 1) * (self.k_h - 1))) // self.stride_h + 1
        
        self.x_addr_base = self.x_in_addr - (self.padding_top * self.x_in_w + self.padding_left) * self.cin_16 * self.x_dw / 256
        self.x_addr_gap_1 = self.stride_w * self.cin_16 * self.x_dw / 256
        # self.x_addr_gap_2 = -(self.xreg_depth - 1) * self.stride_w * self.cin_16 * self.x_dw / 256 + self.xreg_width * self.x_dw / 256
        self.x_addr_gap_2 = 0 + self.xreg_width * self.x_dw / 256
        self.x_addr_gap_3 = 0 - (self.cin_16 / self.xreg_width - 1) * self.xreg_width * self.x_dw / 256 + self.dilation_w * self.cin_16 * self.x_dw / 256
        self.x_addr_gap_4 = 0 - (self.cin_16 / self.xreg_width - 1) * self.xreg_width * self.x_dw / 256 - (self.k_w - 1) * self.dilation_w * self.cin_16 * self.x_dw / 256 + self.dilation_h * self.x_in_w * self.cin_16 * self.x_dw / 256
        self.x_addr_gap_5 = 0 - (self.cin_16 / self.xreg_width - 1) * self.xreg_width * self.x_dw / 256 - (self.k_w - 1) * self.dilation_w * self.cin_16 * self.x_dw / 256 - (self.k_h - 1) * self.dilation_h * self.x_in_w * self.cin_16 * self.x_dw / 256
        self.x_addr_gap_6 = 0 - (self.cin_16 / self.xreg_width - 1) * self.xreg_width * self.x_dw / 256 - (self.k_w - 1) * self.dilation_w * self.cin_16 * self.x_dw / 256 - (self.k_h - 1) * self.dilation_h * self.x_in_w * self.cin_16 * self.x_dw / 256 # + self.xreg_depth * self.stride_w * self.cin_16 * self.x_dw / 256
        # self.x_addr_gap_7 = 0 - (self.cin_16 / self.xreg_width - 1) * self.xreg_width * self.x_dw / 256 - (self.k_w - 1) * self.dilation_w * self.cin_16 * self.x_dw / 256 - (self.k_h - 1) * self.dilation_h * self.x_in_w * self.cin_16 * self.x_dw / 256 - (self.y_out_w / self.xreg_depth - 1) * self.xreg_depth * self.stride_w * self.cin_16 * self.x_dw / 256 + self.stride_h * self.x_in_w * self.cin_16 * self.x_dw / 256
        self.x_addr_gap_7 = 0 - (self.cin_16 / self.xreg_width - 1) * self.xreg_width * self.x_dw / 256 - (self.k_w - 1) * self.dilation_w * self.cin_16 * self.x_dw / 256 - (self.k_h - 1) * self.dilation_h * self.x_in_w * self.cin_16 * self.x_dw / 256 - 0 + self.stride_h * self.x_in_w * self.cin_16 * self.x_dw / 256
        self.x_addr_gap_8 = 0 - (self.cin_16 / self.xreg_width - 1) * self.xreg_width * self.x_dw / 256 - (self.k_w - 1) * self.dilation_w * self.cin_16 * self.x_dw / 256 - (self.k_h - 1) * self.dilation_h * self.x_in_w * self.cin_16 * self.x_dw / 256 - 0 - (self.y_out_h - 1) * self.stride_h * self.x_in_w * self.cin_16 * self.x_dw / 256 + self.x_in_w * self.x_in_h * self.cin_16 * self.x_dw / 256
        self.x_addr_gap_9 = (self.y_out_w - 1) * self.stride_w * self.cin_16 * self.x_dw / 256

        self.w_addr_gap_1 = self.c_out * self.w_dw / 256
        self.w_addr_gap_2 = -0 + self.xreg_width * self.c_out * self.w_dw / 256
        self.w_addr_gap_3 = -0 - self.cin_head * self.xreg_width * self.c_out * self.w_dw / 256 + self.c_in * self.c_out * self.w_dw / 256
        self.w_addr_gap_4 = -0 - self.cin_head * self.xreg_width * self.c_out * self.w_dw / 256 - (self.k_w - 1) * self.c_in * self.c_out * self.w_dw / 256 + self.k_w * self.c_in * self.c_out * self.w_dw / 256
        self.w_addr_gap_5 = -0 - self.cin_head * self.xreg_width * self.c_out * self.w_dw / 256 - (self.k_w - 1) * self.c_in * self.c_out * self.w_dw / 256 - (self.k_h - 1) * self.k_w * self.c_in * self.c_out * self.w_dw / 256 + self.wreg_width * self.w_dw / 256
        self.w_addr_gap_6 = -0 - self.cin_head * self.xreg_width * self.c_out * self.w_dw / 256 - (self.k_w - 1) * self.c_in * self.c_out * self.w_dw / 256 - (self.k_h - 1) * self.k_w * self.c_in * self.c_out * self.w_dw / 256 - (self.c_out / self.wreg_width - 1) * self.wreg_width * self.w_dw / 256

        self.y_addr_gap_0 = 0
        self.y_addr_gap_1 = self.c_out * self.y_dw / 256
        self.y_addr_gap_2 = -self.y_addr_gap_0 - 0 + self.wreg_width * self.y_dw / 256
        self.y_addr_gap_3 = -self.y_addr_gap_0 - 0 - (self.c_out / self.wreg_width - 1) * self.wreg_width * self.y_dw / 256 # + self.xreg_depth * self.c_out * self.y_dw / 256
        # self.y_addr_gap_4 = -self.y_addr_gap_0 - 0 - (self.c_out / self.wreg_width - 1) * self.wreg_width * self.y_dw / 256 - (self.y_out_w / self.xreg_depth - 1) * self.xreg_depth * self.c_out * self.y_dw / 256 + self.y_out_w * self.c_out * self.y_dw / 256
        self.y_addr_gap_4 = -self.y_addr_gap_0 - 0 - (self.c_out / self.wreg_width - 1) * self.wreg_width * self.y_dw / 256 - 0 + self.y_out_w * self.c_out * self.y_dw / 256
        self.y_addr_gap_5 = -self.y_addr_gap_0 - 0 - (self.c_out / self.wreg_width - 1) * self.wreg_width * self.y_dw / 256 - 0 - (self.y_out_h - 1) * self.y_out_w * self.c_out * self.y_dw / 256 + self.y_out_h * self.y_out_w * self.c_out * self.y_dw / 256
        self.y_addr_gap_6 = (self.y_out_w - 1) * self.c_out * self.y_dw / 256

        self.b_addr_gap = self.wreg_width * self.b_dw / 256
        
        self.xw_cnt_gap_1 = self.dilation_w
        self.xw_cnt_gap_2 = -(self.k_w - 1) * self.dilation_w
        self.xw_cnt_gap_3 = -(self.k_w - 1) * self.dilation_w + self.xreg_depth * self.stride_w
        self.xw_cnt_gap_4 = -(self.k_w - 1) * self.dilation_w - ((self.y_out_w - 1) // self.xreg_depth) * self.xreg_depth * self.stride_w
        self.xh_cnt_gap_1 = self.dilation_h
        self.xh_cnt_gap_2 = -(self.k_h - 1) * self.dilation_h
        self.xh_cnt_gap_3 = -(self.k_h - 1) * self.dilation_h + self.stride_h
        self.xh_cnt_gap_4 = -(self.k_h - 1) * self.dilation_h - (self.y_out_h - 1) * self.stride_h

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
        self.param1[208:192] = to_signed_bits(int(self.x_addr_gap_5),16)
        self.param1[224:208] = to_signed_bits(int(self.x_addr_gap_6),16)
        self.param1[240:224] = to_signed_bits(int(self.x_addr_gap_7),16)
        self.param1[256:240] = to_signed_bits(int(self.x_addr_gap_8),16)
        self.param2 = intbv(0)[256:]
        self.param2[16:0] = to_signed_bits(int(self.w_addr_gap_5),16)
        self.param2[32:16] = to_signed_bits(int(self.w_addr_gap_6),16)
        self.param2[48:32] = to_signed_bits(int(self.y_addr_gap_4),16)
        self.param2[61:48] = to_signed_bits(int(self.xw_cnt_gap_1),13)
        self.param2[74:61] = to_signed_bits(int(self.xw_cnt_gap_2),13)
        self.param2[87:74] = to_signed_bits(int(self.xw_cnt_gap_3),13)
        self.param2[100:87] = to_signed_bits(int(self.xw_cnt_gap_4),13)
        self.param2[113:100] = to_signed_bits(int(self.xh_cnt_gap_1),13)
        self.param2[126:113] = to_signed_bits(int(self.xh_cnt_gap_2),13)
        self.param2[139:126] = to_signed_bits(int(self.xh_cnt_gap_3),13)
        self.param2[152:139] = to_signed_bits(int(self.xh_cnt_gap_4),13)
        self.param2[168:152] = to_signed_bits(int(self.y_addr_gap_5),16)
        self.param2[184:168] = to_signed_bits(int(self.x_addr_gap_9),16)
        self.param2[200:184] = to_signed_bits(int(self.y_addr_gap_6),16)

        self.data_blocks["param1"] = DataBlock(data=self.param1, length=1, zero=0, addressing="32B")
        self.data_blocks["param2"] = DataBlock(data=self.param2, length=1, zero=0, addressing="32B")


        self.setAddrBlocks()
        
        self.setPIC()
        
    def setPIC(self):
        self.PIC = intbv(0, min=0, max=(1<<256))
        
        self.PIC[4:0] = 0x1
        self.PIC[8:4] = 0x8
        
        self.PIC[16:8] = self.deps
        self.PIC[32:16] = to_signed_bits(int(self.x_addr_base),16)
        self.PIC[47:32] = self.w_in_addr
        self.PIC[62:47] = self.b_in_addr
        self.PIC[77:62] = self.y_out_addr
        self.PIC[92:77] = self.param_addr_1
        self.PIC[104:92] = self.x_in_w + self.padding_left
        self.PIC[116:104] = self.x_in_h + self.padding_top
        self.PIC[124:116] = self.padding_left
        self.PIC[132:124] = self.padding_top
        self.PIC[144:138] = self.c_out / self.wreg_width - 1
        self.PIC[148:144] = self.k_w - 1
        self.PIC[152:148] = self.k_h - 1
        if (self.y_out_w % self.xreg_depth == 0):
            self.PIC[160:152] = self.y_out_w // self.xreg_depth - 1
        else:
            self.PIC[160:152] = self.y_out_w // self.xreg_depth
        self.PIC[172:160] = self.y_out_h - 1
        self.PIC[182:172] = self.bs - 1
        self.PIC[198:182] = to_bf16_bits(float(self.padding_value))
        self.PIC[213:198] = self.param_addr_2
        self.PIC[218:213] = self.y_out_w % self.xreg_depth
        if (self.c_in % self.xreg_width == 0):
            self.PIC[138:132] = self.c_in // self.xreg_width - 1
        else:
            self.PIC[138:132] = self.c_in // self.xreg_width
        self.PIC[225:218] = self.c_in % self.xreg_width
        
    def setAddrBlocks(self):
        pass
    
    def handwritten_bf16_conv(self, padded_x: torch.Tensor,w: torch.Tensor,b: torch.Tensor,y: torch.Tensor, sh, sw, dh, dw):
        assert padded_x.dtype == torch.bfloat16
        assert w.dtype == torch.bfloat16
        assert b.dtype == torch.bfloat16
        assert y.dtype == torch.bfloat16

        batch, cin, xh, xw = padded_x.shape
        cout, _, kh, kw = w.shape
        _, _, yh, yw = y.shape

        assert (cin % 16) == 0, "cin % 16 != 0"
        assert (cout % 16) == 0, "cout % 16 != 0"

        cinloop = int(cin / 16)
        coutloop = int(cout / 16)
        ywres = yw % 8
        if (ywres == 0):
            ywloop = yw // 8
        else:
            ywloop = yw // 8 + 1

        for b0 in range(batch):
            for yh0 in range(yh):
                for ywl0 in range(ywloop):
                    for coutloop0 in range(coutloop):
                        for i in range(16):
                            if (ywl0 < (ywloop - 1) or ywres == 0):
                                for j in range(8):
                                    y[b0, coutloop0*16+i, yh0, ywl0*8+j] = b[coutloop0*16+i]
                            else:
                                for j in range(ywres):
                                    y[b0, coutloop0*16+i, yh0, ywl0*8+j] = b[coutloop0*16+i]
                        for kh0 in range(kh):
                            for kw0 in range(kw):
                                for cinloop0 in range(cinloop):
                                    for i in range(16):
                                        for k in range(16):
                                            if (ywl0 < (ywloop - 1) or ywres == 0):
                                                for j in range(8):
                                                    y[b0, coutloop0*16+i, yh0, ywl0*8+j] += padded_x[b0, cinloop0*16+k, yh0*dh+kh0*sh, (ywl0*8+j)*dw+kw0*sw] * w[coutloop0*16+i, cinloop0*16+k, kh0, kw0]
                                            else:
                                                for j in range(ywres):
                                                    y[b0, coutloop0*16+i, yh0, ywl0*8+j] += padded_x[b0, cinloop0*16+k, yh0*dh+kh0*sh, (ywl0*8+j)*dw+kw0*sw] * w[coutloop0*16+i, cinloop0*16+k, kh0, kw0]

        return y

    def execute(self, core: EmuCore):
        # Get the input x tensor
        x_data_length = core.memory.getTensorLen((self.bs, self.c_in, self.x_in_h, self.x_in_w), torch.bfloat16, (0, 2, 3, 1))
        x_tensor = core.memory.readTensor(self.x_in_addr, x_data_length, (self.bs, self.c_in, self.x_in_h, self.x_in_w), torch.bfloat16, (0, 2, 3, 1))
        # Get the weight w tensor
        w_data_length = core.memory.getTensorLen((self.c_out, self.c_in, self.k_h, self.k_w), torch.bfloat16, (2, 3, 1, 0))
        w_tensor = core.memory.readTensor(self.w_in_addr, w_data_length, (self.c_out, self.c_in, self.k_h, self.k_w), torch.bfloat16, (2, 3, 1, 0))
        # Get the bias b tensor
        b_data_length = core.memory.getTensorLen((self.c_out,), torch.bfloat16, (0,))
        b_tensor = core.memory.readTensor(self.b_in_addr, b_data_length, (self.c_out,), torch.bfloat16)
        # Pad the input x tensor
        padded_x_in = F.pad(x_tensor,(self.padding_left,self.padding_right,self.padding_top,self.padding_bottom),value=self.padding_value)
        # Create the output y tensor
        y_tensor = torch.zeros((self.bs, self.c_out, self.y_out_h, self.y_out_w), dtype=torch.bfloat16)
        # Calculate the output y tensor
        y_tensor = self.handwritten_bf16_conv(padded_x_in, w_tensor, b_tensor, y_tensor, self.stride_h, self.stride_w, self.dilation_h, self.dilation_w)
        # Write the output y tensor back to memory
        core.memory.writeTensor(self.y_out_addr, y_tensor, (0, 2, 3, 1))
    
    def generate_events(self, core: AnalyCore):
        events = []
        L6_max = self.bs
        L5_max = self.y_out_h
        if self.conv_type == 'INT8':
            L4_max = math.ceil(self.y_out_w / core.config["core"]["int8_PE_array_height"])
            L3_max = math.ceil(self.c_out // core.config["core"]["int8_PE_array_width"])
            L0_max = math.ceil(self.c_in / core.config["core"]["int8_PE_buffer_depth"])
        elif self.conv_type == 'BF16':
            L4_max = math.ceil(self.y_out_w / core.config["core"]["bf16_PE_array_height"])
            L3_max = math.ceil(self.c_out / core.config["core"]["bf16_PE_array_width"])
            L0_max = math.ceil(self.c_in / core.config["core"]["bf16_PE_buffer_depth"])
        L2_max = self.k_h
        L1_max = self.k_w
        # 一次循环的计算次数
        inner_compute_cycle = core.config["core"]["int8_PE_buffer_depth"] if self.conv_type == 'INT8' else core.config["core"]["bf16_PE_buffer_depth"]
        # 一次循环需要读取的元素数量
        inner_input_num = (core.config["core"]["int8_PE_buffer_depth"] if self.conv_type == 'INT8' else core.config["core"]["bf16_PE_buffer_depth"]) * core.config["core"]["bf16_PE_array_height"]
        inner_weight_num = (core.config["core"]["int8_PE_buffer_depth"] if self.conv_type == 'INT8' else core.config["core"]["bf16_PE_buffer_depth"]) * core.config["core"]["bf16_PE_array_width"]
        # 输出一次需要读取的偏置和写入的输出数量
        per_loop_bias_num = core.config["core"]["int8_PE_array_width"] if self.conv_type == 'INT8' else core.config["core"]["bf16_PE_array_width"]
        per_loop_output_num = (core.config["core"]["int8_PE_array_height"] * core.config["core"]["int8_PE_array_width"] if self.conv_type == 'INT8' else core.config["core"]["bf16_PE_array_height"] * core.config["core"]["bf16_PE_array_width"])
        # 一次循环读取输入的时间（读要两个时钟，写只要一个时钟）
        # if torch.bfloat16 == torch.bool:
        #     inner_input_cycle = 0
        #     snn_input_cycle = (core.config["core"]["int8_PE_array_height"] if self.conv_type == 'INT8' else core.config["core"]["bf16_PE_array_height"]) * math.ceil(self.c_in * get_byte_num(torch.bfloat16) / core.config["core"]["memory_width"])
        # else:
        #     inner_input_cycle = math.ceil(inner_input_num * get_byte_num(torch.bfloat16) / core.config["core"]["memory_width"]) * 2
        inner_input_cycle = math.ceil(inner_input_num * get_byte_num(torch.bfloat16) / core.config["core"]["memory_width"]) * 2
        # 一次循环读取权重的时间（读要两个时钟，写只要一个时钟）
        inner_weight_cycle = math.ceil(inner_weight_num * get_byte_num(torch.bfloat16) / core.config["core"]["memory_width"]) * 2
        # 读取偏置和写入输出的时间（读要两个时钟，写只要一个时钟）
        per_loop_bias_cycle = math.ceil(per_loop_bias_num * get_byte_num(torch.bfloat16) / core.config["core"]["memory_width"]) * 2
        per_loop_output_cycle = math.ceil(per_loop_output_num * get_byte_num(torch.bfloat16) / core.config["core"]["memory_width"])
        # 最后一个cell的x和第一个cell的w并行读取，mac计算结束比读w晚一个时钟
        inner_cycle = inner_input_cycle + max(inner_weight_cycle - 1, inner_compute_cycle)
        # 计算总时钟数
        total_cycle = L6_max * L5_max * L4_max * L3_max * L2_max * L1_max * L0_max * inner_cycle + L6_max * L5_max * L4_max * L3_max * per_loop_bias_cycle + L6_max * L5_max * L4_max * L3_max * per_loop_output_cycle # 精度转换一个时钟（未计入）
        # 计算总计算量
        total_computation = L6_max * L5_max * L4_max * L3_max * L2_max * L1_max * L0_max * inner_compute_cycle * (core.config["core"]["int8_PE_array_height"] * core.config["core"]["int8_PE_array_width"] if self.conv_type == 'INT8' else core.config["core"]["bf16_PE_array_height"] * core.config["core"]["bf16_PE_array_width"])
        # 计算能耗
        if self.conv_type == 'INT8':
            if torch.bfloat16 == torch.bool:
                energy = total_computation * core.config["core"]["int8_PE_add_energy"]
            else:
                energy = total_computation * (core.config["core"]["int8_PE_energy"])
        elif self.conv_type == 'BF16':
            if torch.bfloat16 == torch.bool:
                energy = total_computation * core.config["core"]["bf16_PE_add_energy"]
            else:
                energy = total_computation * (core.config["core"]["bf16_PE_energy"])
        # compute event
        compute_event = ComputeEvent(name="Conv", 
                                     parent=core.matrix.full_name, 
                                     compute_type=EventType.MATRIX, 
                                     computation=total_computation, 
                                     theoretical_computation=self.c_out * self.y_out_h * self.y_out_w * self.bs * self.c_in * self.k_h * self.k_w, 
                                     max_consume_rate=total_computation / total_cycle, 
                                     energy=energy)
        events.append(compute_event)
        # 暂时只考虑单级缓存
        # read event
        L0_volume = 0
        # read input
        L0_volume += inner_input_num * get_byte_num(torch.bfloat16) * L0_max * L1_max * L2_max * L3_max * L4_max * L5_max * L6_max
        # read weight
        L0_volume += inner_weight_num * get_byte_num(torch.bfloat16) * L0_max * L1_max * L2_max * L3_max * L4_max * L5_max * L6_max
        # read bias
        L0_volume += per_loop_bias_num * get_byte_num(torch.bfloat16) * L3_max * L4_max * L5_max * L6_max
        L0_input_event = MemoryEvent(name="Conv_input", 
                                         parent=core.memory[0].full_name, 
                                         memory_type=EventType.READ, 
                                         volume=L0_volume, 
                                         bounded_events=[], 
                                         energy=math.ceil(L0_volume / core.config["core"]["memory_width"]) * core.config["core"]["L0_memory_read_energy"], 
                                         max_bandwidth=L0_volume / total_cycle,
                                         hierarchy=0)
        events.append(L0_input_event)
        # write event
        output_volume = per_loop_output_num * get_byte_num(torch.bfloat16) * L3_max * L4_max * L5_max * L6_max
        output_event = MemoryEvent(name="Conv_output",
                                   parent=core.memory[0].full_name,
                                   memory_type=EventType.WRITE,
                                   volume=output_volume,
                                   bounded_events=[],
                                   energy=(math.ceil(output_volume / core.config["core"]["memory_width"]) * core.config["core"]["L0_memory_write_energy"]),
                                   max_bandwidth=output_volume / total_cycle,
                                   hierarchy=0)
        events.append(output_event)
        
        event_nms = []
        for event in events:
            event_nms.append(event.full_name)
        for event in events:
            if event.event_type in [EventType.READ, EventType.INOUT, EventType.WRITE]:
                event.bounded_events = event_nms
                event.bounded_events.remove(event.full_name)

        return events