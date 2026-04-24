from .prim import Primitive
from basics.addr_blocks_in_mem import AddrBlocksInMem
from basics.data_block import DataBlock
from core.simulator.emulator.core import Core as EmuCore
from core.simulator.analyser.core import Core as AnalyCore

from copy import deepcopy
from myhdl import bin, intbv

class PrimBar(Primitive):
    def __init__(self, deps, delay, adaptive):
        super().__init__()
        self.name = "Bar"
        
        self.deps = deps
        self.delay = delay
        self.adaptive = adaptive
        
        # 生成PIC
        self.setPIC()
        
    def setPIC(self):
        self.PIC = intbv(0, min=0, max=(1<<256))
        self.PIC[4:0] = 0x0
        self.PIC[8:4] = 0x4
        self.PIC[16:8] = self.deps
        self.PIC[48:32] = self.delay
        self.PIC[81:80] = self.adaptive
        
    def setAddrBlocks(self):
        pass
    
    def execute(self, core: EmuCore):
        # TODO
        pass
    
    def generate_events(self, core: AnalyCore):
        events = []
        # TODO
        return events