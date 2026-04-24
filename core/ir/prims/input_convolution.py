from core.ir.prims.prim import Primitive
from basics.addr_blocks_in_mem import AddrBlocksInMem
from basics.data_block import DataBlock
from core.simulator.emulator.core import Core as EmuCore
from core.simulator.analyser.core import Core as AnalyCore

from copy import deepcopy
from myhdl import bin, intbv
import math, struct

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

class PrimInputConvolution(Primitive):
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
        b_in_data,
        x_in_data,
        w_in_data):
        super().__init__()
        self.name = "InputConvolution"  # 定义名称
        self.conv_type = conv_type
        self.isSNN = isSNN
        if self.conv_type == 'INT8':
            self.w_dw = 8
            if self.isSNN:
                self.x_dw = 1
            else:
                self.x_dw = 8
            self.b_dw = 32
            self.y_dw = 16
            self.reg_depth = 16
        elif self.conv_type == 'BF16':
            self.w_dw = 16
            if self.isSNN:
                self.x_dw = 1
            else:
                self.x_dw = 16
            self.b_dw = 16
            self.y_dw = 16
            self.reg_depth = 8
        self.array_col = 256 / self.w_dw
        self.array_row = 256 / self.x_dw
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

        self.xw_pad = self.x_in_w + self.padding_left + self.padding_right
        self.xh_pad = self.x_in_h + self.padding_top + self.padding_bottom
        self.y_out_w = (self.x_in_w + self.padding_left + self.padding_right - (self.k_w + (self.dilation_w - 1) * (self.k_w - 1))) // self.stride_w + 1
        self.y_out_h = (self.x_in_h + self.padding_top + self.padding_bottom - (self.k_h + (self.dilation_h - 1) * (self.k_h - 1))) // self.stride_h + 1
        self.send_cell_num_1 = int(self.c_out * self.b_dw / 256)
        self.send_cell_num_2 = int(self.bs * self.c_in * self.x_in_w * self.x_in_h * self.x_dw / 256)
        self.send_cell_num_3 = int(self.c_in * self.c_out * self.k_w * self.k_h * self.w_dw / 256)
        self.send_cell_num_4 = int(self.bs * self.c_out * self.y_out_w * self.y_out_h * self.y_dw / 256)

        self.data_blocks["x_in_data"] = deepcopy(x_in_data)
        self.data_blocks["w_in_data"] = deepcopy(w_in_data)
        self.data_blocks["b_in_data"] = deepcopy(b_in_data)

        self.x_addr_base = self.x_in_addr - (self.padding_top * self.x_in_w + self.padding_left) * self.c_in * self.x_dw / 256
        self.x_addr_gap_1 = self.stride_w * self.c_in * self.x_dw / 256
        # self.x_addr_gap_2 = -(self.reg_depth - 1) * self.stride_w * self.c_in * self.x_dw / 256 + self.array_row * self.x_dw / 256
        self.x_addr_gap_2 = 0 + self.array_row * self.x_dw / 256
        self.x_addr_gap_3 = 0 - (self.c_in / self.array_row - 1) * self.array_row * self.x_dw / 256 + self.dilation_w * self.c_in * self.x_dw / 256
        self.x_addr_gap_4 = 0 - (self.c_in / self.array_row - 1) * self.array_row * self.x_dw / 256 - (self.k_w - 1) * self.dilation_w * self.c_in * self.x_dw / 256 + self.dilation_h * self.x_in_w * self.c_in * self.x_dw / 256
        self.x_addr_gap_5 = 0 - (self.c_in / self.array_row - 1) * self.array_row * self.x_dw / 256 - (self.k_w - 1) * self.dilation_w * self.c_in * self.x_dw / 256 - (self.k_h - 1) * self.dilation_h * self.x_in_w * self.c_in * self.x_dw / 256
        self.x_addr_gap_6 = 0 - (self.c_in / self.array_row - 1) * self.array_row * self.x_dw / 256 - (self.k_w - 1) * self.dilation_w * self.c_in * self.x_dw / 256 - (self.k_h - 1) * self.dilation_h * self.x_in_w * self.c_in * self.x_dw / 256 # + self.reg_depth * self.stride_w * self.c_in * self.x_dw / 256
        # self.x_addr_gap_7 = 0 - (self.c_in / self.array_row - 1) * self.array_row * self.x_dw / 256 - (self.k_w - 1) * self.dilation_w * self.c_in * self.x_dw / 256 - (self.k_h - 1) * self.dilation_h * self.x_in_w * self.c_in * self.x_dw / 256 - (self.y_out_w / self.reg_depth - 1) * self.reg_depth * self.stride_w * self.c_in * self.x_dw / 256 + self.stride_h * self.x_in_w * self.c_in * self.x_dw / 256
        self.x_addr_gap_7 = 0 - (self.c_in / self.array_row - 1) * self.array_row * self.x_dw / 256 - (self.k_w - 1) * self.dilation_w * self.c_in * self.x_dw / 256 - (self.k_h - 1) * self.dilation_h * self.x_in_w * self.c_in * self.x_dw / 256 - 0 + self.stride_h * self.x_in_w * self.c_in * self.x_dw / 256
        self.x_addr_gap_8 = 0 - (self.c_in / self.array_row - 1) * self.array_row * self.x_dw / 256 - (self.k_w - 1) * self.dilation_w * self.c_in * self.x_dw / 256 - (self.k_h - 1) * self.dilation_h * self.x_in_w * self.c_in * self.x_dw / 256 - 0 - (self.y_out_h - 1) * self.stride_h * self.x_in_w * self.c_in * self.x_dw / 256 + self.x_in_w * self.x_in_h * self.c_in * self.x_dw / 256
        self.x_addr_gap_9 = (self.y_out_w - 1) * self.stride_w * self.c_in * self.x_dw / 256

        self.w_addr_gap_1 = self.c_out * self.w_dw / 256
        self.w_addr_gap_2 = -(self.array_row - 1) * self.c_out * self.w_dw / 256 + self.array_row * self.c_out * self.w_dw / 256
        self.w_addr_gap_3 = -(self.array_row - 1) * self.c_out * self.w_dw / 256 - (self.c_in / self.array_row - 1) * self.array_row * self.c_out * self.w_dw / 256 + self.c_in * self.c_out * self.w_dw / 256
        self.w_addr_gap_4 = -(self.array_row - 1) * self.c_out * self.w_dw / 256 - (self.c_in / self.array_row - 1) * self.array_row * self.c_out * self.w_dw / 256 - (self.k_w - 1) * self.c_in * self.c_out * self.w_dw / 256 + self.k_w * self.c_in * self.c_out * self.w_dw / 256
        self.w_addr_gap_5 = -(self.array_row - 1) * self.c_out * self.w_dw / 256 - (self.c_in / self.array_row - 1) * self.array_row * self.c_out * self.w_dw / 256 - (self.k_w - 1) * self.c_in * self.c_out * self.w_dw / 256 - (self.k_h - 1) * self.k_w * self.c_in * self.c_out * self.w_dw / 256 + self.array_col * self.w_dw / 256
        self.w_addr_gap_6 = -(self.array_row - 1) * self.c_out * self.w_dw / 256 - (self.c_in / self.array_row - 1) * self.array_row * self.c_out * self.w_dw / 256 - (self.k_w - 1) * self.c_in * self.c_out * self.w_dw / 256 - (self.k_h - 1) * self.k_w * self.c_in * self.c_out * self.w_dw / 256 - (self.c_out / self.array_col - 1) * self.array_col * self.w_dw / 256

        if self.conv_type == 'INT8':
            self.y_addr_gap_0 = 1
        elif self.conv_type == 'BF16':
            self.y_addr_gap_0 = 0
        self.y_addr_gap_1 = self.c_out * self.y_dw / 256
        self.y_addr_gap_2 = -self.y_addr_gap_0 - 0 + self.array_col * self.y_dw / 256
        self.y_addr_gap_3 = -self.y_addr_gap_0 - 0 - (self.c_out / self.array_col - 1) * self.array_col * self.y_dw / 256 # + self.reg_depth * self.c_out * self.y_dw / 256
        # self.y_addr_gap_4 = -self.y_addr_gap_0 - 0 - (self.c_out / self.array_col - 1) * self.array_col * self.y_dw / 256 - (self.y_out_w / self.reg_depth - 1) * self.reg_depth * self.c_out * self.y_dw / 256 + self.y_out_w * self.c_out * self.y_dw / 256
        self.y_addr_gap_4 = -self.y_addr_gap_0 - 0 - (self.c_out / self.array_col - 1) * self.array_col * self.y_dw / 256 - 0 + self.y_out_w * self.c_out * self.y_dw / 256
        self.y_addr_gap_5 = -self.y_addr_gap_0 - 0 - (self.c_out / self.array_col - 1) * self.array_col * self.y_dw / 256 - 0 - (self.y_out_h - 1) * self.y_out_w * self.c_out * self.y_dw / 256 + self.y_out_h * self.y_out_w * self.c_out * self.y_dw / 256
        self.y_addr_gap_6 = (self.y_out_w - 1) * self.c_out * self.y_dw / 256

        self.b_addr_gap = self.array_col * self.b_dw / 256
        
        self.xw_cnt_gap_1 = self.dilation_w
        self.xw_cnt_gap_2 = -(self.k_w - 1) * self.dilation_w
        self.xw_cnt_gap_3 = -(self.k_w - 1) * self.dilation_w + self.reg_depth * self.stride_w
        self.xw_cnt_gap_4 = -(self.k_w - 1) * self.dilation_w - ((self.y_out_w - 1) // self.reg_depth) * self.reg_depth * self.stride_w
        self.xh_cnt_gap_1 = self.dilation_h
        self.xh_cnt_gap_2 = -(self.k_h - 1) * self.dilation_h
        self.xh_cnt_gap_3 = -(self.k_h - 1) * self.dilation_h + self.stride_h
        self.xh_cnt_gap_4 = -(self.k_h - 1) * self.dilation_h - (self.y_out_h - 1) * self.stride_h

        # print(self.x_addr_gap_1,self.x_addr_gap_2,self.x_addr_gap_3,self.x_addr_gap_4,self.x_addr_gap_5,self.x_addr_gap_6,self.x_addr_gap_7,self.x_addr_gap_8)
        # print(self.w_addr_gap_1,self.w_addr_gap_2,self.w_addr_gap_3,self.w_addr_gap_4,self.w_addr_gap_5,self.w_addr_gap_6)
        # print(self.y_addr_gap_1,self.y_addr_gap_2,self.y_addr_gap_3,self.y_addr_gap_4)

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


        # 计算地址
        self.setAddrBlocks()
        
        # 生成PIC
        self.setPIC()
        
        # 检查数据
        self.check_data()
        
    def setPIC(self):
        self.PIC = intbv(0, min=0, max=(1<<256))
        
        self.PIC[4:0] = 0x1
        if self.conv_type == 'INT8' and self.isSNN == 0:
            self.PIC[8:4] = 0x0
        elif self.conv_type == 'BF16' and self.isSNN == 0:
            self.PIC[8:4] = 0x1
        elif self.conv_type == 'INT8' and self.isSNN == 1:
            self.PIC[8:4] = 0x2
        elif self.conv_type == 'BF16' and self.isSNN == 1:
            self.PIC[8:4] = 0x3
        
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
        self.PIC[138:132] = self.c_in / self.array_row - 1
        self.PIC[144:138] = self.c_in / self.array_col - 1
        self.PIC[148:144] = self.k_w - 1
        self.PIC[152:148] = self.k_h - 1
        if (self.y_out_w % self.reg_depth == 0):
            self.PIC[160:152] = self.y_out_w // self.reg_depth - 1
        else:
            self.PIC[160:152] = self.y_out_w // self.reg_depth
        self.PIC[172:160] = self.y_out_h - 1
        self.PIC[182:172] = self.bs - 1
        if self.conv_type == 'INT8':
            self.PIC[190:182] = to_signed_bits(int(self.padding_value),8)
        elif self.conv_type == 'BF16':
            self.PIC[198:182] = to_bf16_bits(float(self.padding_value))
            # print(self.PIC[198:182])
        self.PIC[213:198] = self.param_addr_2
        self.PIC[218:213] = self.y_out_w % self.reg_depth
        
    def setAddrBlocks(self):
        self.data_addr_list["b_in_data"] = AddrBlocksInMem({self.b_in_addr:self.send_cell_num_1}, "32B")
        self.data_addr_list["x_in_data"] = AddrBlocksInMem({self.x_in_addr:self.send_cell_num_2}, "32B")
        self.data_addr_list["w_in_data"] = AddrBlocksInMem({self.w_in_addr:self.send_cell_num_3}, "32B")
        self.data_addr_list["y_out_data"] = AddrBlocksInMem({self.y_out_addr:self.send_cell_num_4}, "32B")
        self.data_addr_list["param1"] = AddrBlocksInMem({self.param_addr_1: 1}, "32B")
        self.data_addr_list["param2"] = AddrBlocksInMem({self.param_addr_2: 1}, "32B")
        pass
    
    def execute(self, core: EmuCore):
        # TODO
        pass
    
    def generate_events(self, core: AnalyCore):
        events = []
        # TODO
        return events
