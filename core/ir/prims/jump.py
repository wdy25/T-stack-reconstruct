from core.ir.operations import jump
from core.ir.prims.prim import Primitive
from basics.addr_blocks_in_mem import AddrBlocksInMem
from basics.data_block import DataBlock

from copy import deepcopy
from myhdl import bin, intbv

class PrimJump(Primitive):
    def __init__(self, deps, jump_addr, relative):
        super().__init__()
        self.name = "Jump"  # 定义名称
        
        self.deps = deps
        self.jump_addr = jump_addr
        self.relative = relative
        
        # 生成PIC
        self.setPIC()
        
    def setPIC(self):
        self.PIC = intbv(0, min=0, max=(1<<256))
        self.PIC[4:0] = 0x0
        self.PIC[8:4] = 0x0
        self.PIC[16:8] = self.deps
        self.PIC[47:32] = intbv(self.jump_addr, min=-(1<<14), max=(1<<14))[15:0]
        self.PIC[81:80] = self.relative
        
    def setAddrBlocks(self):
        pass

    def execute(self, core):

        if self.relative == 1 :
            core.PC += self.jump_addr
        else:
            core.PC = self.jump_addr
    
    def generate_events(self, core):
        return []