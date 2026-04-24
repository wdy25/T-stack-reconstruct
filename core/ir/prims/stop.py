from core.ir.prims.prim import Primitive
from basics.addr_blocks_in_mem import AddrBlocksInMem
from basics.data_block import DataBlock
from core.simulator.emulator.core import Core as EmuCore
from core.simulator.analyser.core import Core as AnalyCore

from copy import deepcopy
from myhdl import bin, intbv

class PrimStop(Primitive):
    def __init__(self, deps, jump_addr, jump, relative):
        super().__init__()
        self.name = "Stop"
        
        self.deps = deps
        self.jump_addr = jump_addr
        self.jump = jump
        self.relative = relative
        
        # 生成PIC
        self.setPIC()
        
    def setPIC(self):
        self.PIC = intbv(0, min=0, max=(1<<256))
        self.PIC[4:0] = 0x0
        self.PIC[8:4] = 0x3
        self.PIC[16:8] = self.deps
        self.PIC[47:32] = self.jump_addr
        self.PIC[81:80] = self.relative
        self.PIC[82:81] = self.jump
        
    def execute(self, core: EmuCore):
        core.stop = True
        if self.jump:
            if self.relative:
                core.PC = core.PC + self.jump_addr
            else:
                core.PC = self.jump_addr
        else:
            core.PC = core.PC + 1
    
    def generate_events(self, core: AnalyCore):
        return []