from core.ir.prims.prim import Primitive
from basics.addr_blocks_in_mem import AddrBlocksInMem
from basics.data_block import DataBlock

from copy import deepcopy
from myhdl import bin, intbv

class PrimLoop(Primitive):
    def __init__(self, deps, jump_addr, config_addr, loop_max, relative, loop_cnt: DataBlock=None):
        super().__init__()
        self.name = "Loop"
        
        self.deps = deps
        self.jump_addr = jump_addr
        self.config_addr = config_addr
        self.loop_max = loop_max
        self.relative = relative

        loop_count = intbv(loop_cnt, min=0, max=(1<<256))
        self.data_blocks["loop_count"] = DataBlock(data=loop_count, length=1, zero=0, addressing="32B")        
        
        # 生成PIC
        self.setPIC()
        
    def setPIC(self):
        self.PIC = intbv(0, min=0, max=(1<<256))
        addr_temp = intbv(0, min=-(1<<14), max=(1<<14)-1)
        addr_temp[:] = self.jump_addr
        self.PIC[4:0] = 0x0
        self.PIC[8:4] = 0x2
        self.PIC[16:8] = self.deps
        self.PIC[47:32] = addr_temp[15:0]
        self.PIC[63:48] = self.config_addr
        self.PIC[80:64] = self.loop_max
        self.PIC[81:80] = self.relative
        
    def setAddrBlocks(self):
        self.data_addr_list["loop_count"] = AddrBlocksInMem(data={self.config_addr: 1}, addressing="32B")
    
    def execute(self, core):
        
        loop_count = core.memory[self.config_addr]
        
        if loop_count < self.loop_max:
        
            if self.relative == 1 :
                core.PC += self.jump_addr

            else:
                core.PC = self.jump_addr

            core.memory.writeTensor(self.config_addr, loop_count)

        else:
            core.PC += 1
            
    def generate_events(self, core):
        return []