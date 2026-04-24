from core.ir.prims.prim import Primitive
from basics.addr_blocks_in_mem import AddrBlocksInMem
from basics.data_block import DataBlock

from copy import deepcopy
from myhdl import bin, intbv

class PrimBranch(Primitive):
    def __init__(self, deps, jump_addr, condition_addr, relative, condition_data: DataBlock=None):
        super().__init__()
        self.name = "Branch"
        
        self.deps = deps
        self.jump_addr = jump_addr
        self.condition_addr = condition_addr
        self.relative = relative
        
        # condition = intbv(condition_data, min=0, max=(1<<256))
        # self.data_blocks["condition"] = DataBlock(data=condition, length=1, zero=0, addressing="32B")    
        
        # 生成PIC
        self.setPIC()
        
    def setPIC(self):
        self.PIC = intbv(0, min=0, max=(1<<256))
        self.PIC[4:0] = 0x0
        self.PIC[8:4] = 0x1
        self.PIC[16:8] = self.deps
        self.PIC[47:32] = self.jump_addr
        self.PIC[63:48] = self.condition_addr
        self.PIC[81:80] = self.relative
        
    def setAddrBlocks(self):
        self.data_addr_list["condition"] = AddrBlocksInMem({self.condition_addr: 1}, "32B")

    def execute(self, core):
        
        condition = core.memory[self.condition_addr[0:14]]

        if condition[0:8] > 0:  # TODO: > or >= ?

            if self.relative == 1 :
                core.PC += self.jump_addr
            else:
                core.PC = self.jump_addr
                
        else:
            core.PC += 1

    def generate_events(self, core):
        return []