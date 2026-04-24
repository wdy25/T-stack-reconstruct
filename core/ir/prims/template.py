from core.ir.prims.prim import Primitive
from basics.addr_blocks_in_mem import AddrBlocksInMem
from basics.data_block import DataBlock

from copy import deepcopy
from myhdl import bin, intbv

class PrimTemplate(Primitive):
    def __init__(self, deps, addr0, addr1, param0, param1, param2=0, data0: DataBlock=None, data1: DataBlock=None):
        super().__init__()
        self.name = "Template"  # 定义名称
        
        # 记录参数，即原语中的所有字段，包括para中某些需要额外配置的字段
        self.deps = deps
        self.addr0 = addr0
        self.addr1 = addr1
        self.param0 = param0
        self.param1 = param1
        self.param2 = param2
        
        # 保存原语需要提前配置的数据，如果有para则必须在此生成
        # 对于非para的数据，可以有3种情况：
        # 1. 为None，表示不需要提前配置，可能是其他原语的输出作为输入，直接跳过这份数据即可
        # 2. 为[]，表示需要提前配置，需要生成一个DataBlock，长度根据原语参数确定，zero=0表示全随机
        # 3. 为DataBlock，表示直接使用这个DataBlock 
        if data0 is not None:  # 如果data0为None，表示不需要提前配置，不需要处理
            if data0 == []:
                self.data_blocks["data0"] = DataBlock(data=None, length=param0, zero=0, addressing="32B")
            else:
                self.data_blocks["data0"] = deepcopy(data0)
        
        if data1 is not None:
            if data1 == []:
                self.data_blocks["data1"] = DataBlock(data=None, length=param1, zero=0, addressing="32B")
            else:
                self.data_blocks["data1"] = deepcopy(data1)
        
        
        # 生成para
        para = intbv(0, min=0, max=(1<<256))
        para[4:0] = intbv(param0 * param2, min=0, max=(1<<4))[4:0]
        para[8:4] = intbv(param1, min=-(1<<3), max=(1<<3))[4:0]
        self.data_blocks["para"] = DataBlock(data=para, length=1, zero=0, addressing="32B")
        
        
        # 计算地址 起始地址和数据长度
        self.setAddrBlocks()
        
        # 生成PIC
        self.setPIC()
        
        # 检查数据
        self.check_data()
        
    def setPIC(self):
        self.PIC = intbv(0, min=0, max=(1<<256))
        
        self.PIC[4:0] = 0x0
        self.PIC[8:4] = 0x0
        
        self.PIC[16:8] = self.deps
        
    def setAddrBlocks(self):
        # 除了各种输入数据外，还需要记录结果数据的地址、Para的地址等各种运行中需要的数据地址核长度
        # 此函数不应该直接使用self.data_blocks中的数据长度，因为这里计算的长度需要和self.data_blocks中的数据长度进行对比以确保传入的数据长度是正确的
        
        # self.data_addr_list["data_name"] = AddrBlocksInMem({addr: length}, addressing)
        self.data_addr_list["data0"] = AddrBlocksInMem({self.addr0: self.param0}, "32B")
        self.data_addr_list["data1"] = AddrBlocksInMem({self.addr1: self.param1}, "32B")
        self.data_addr_list["result"] = AddrBlocksInMem({self.addr0 + self.param2: self.param1}, "32B")
        
    def run(self):
        # 这里可以实现原语的具体逻辑
        # 例如，需要提前配置data0和data1等输入data_blocks，进行计算，然后生成输出data_blocks
        pass