from __future__ import annotations

from core.ir.prims.prim import PrimitiveType
from .memory import Memory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .router import NOC

class Core():
    def __init__(self, config, pos_y, pos_x, noc: NOC):
        self.config = config
        self.pos_y = pos_y
        self.pos_x = pos_x
        
        self.noc: NOC = noc
        self.noc.add_core(self)
        
        self.memory: Memory = Memory(config["core"]["memory_capacity_per_core"], config["core"]["memory_width"])
        self.PC = 0
        self.prims = []
        
        self.stop = False
        self.receiving = False
        self.sending = False
        
        self.tag = None
        self.recv_addr = None
        self.recv_num = None
        self.CXY = 0
        self.mc_y = 0
        self.mc_x = 0

    
    def from_start(self):
        self.PC = 0
        self.stop = False
        self.receiving = False
        self.sending = False
        self.tag = None
        self.recv_addr = None
        self.recv_num = None
        
        for prim in self.prims:
            if prim.type == PrimitiveType.ROUTER:
                prim.reset()
    

    def run(self, from_start=False):
        if from_start:
            self.from_start()
        prim_count = 0
        while True:
            if self.stop or self.PC >= len(self.prims):
                break
            
            if prim_count >= self.config["emulator"]["max_prims_per_run"]:
                self.stop = True
                break
            
            # If the core is receiving (waiting), the PC has be updated, so it can not execute until receiving is done
            # but if the core is sending (waiting), the PC has not been updated, so it should execute send again
            if self.receiving:
                break
            
            prim = self.prims[self.PC]
            # print(prim.__class__.__name__)
            prim.execute(self)
            
            if self.stop:
                break
            
            # If the core is sending (waiting), then next time it will still execute send
            if prim.type == PrimitiveType.ROUTER and (self.sending):
                break
            
            if prim.type != PrimitiveType.CONTROL:
                self.PC += 1
            
            # sending means execution is not valid
            prim_count += 1
            
            # If the core is receiving (waiting), then next time it will execute next prim
            if prim.type == PrimitiveType.ROUTER and (self.receiving):
                break
            
            if prim_count >= self.config["emulator"]["max_prims_per_run"]:
                self.stop = True
                break
        
        return prim_count
    
    
    '''
    prims: list of tuples, each tuple contains a primitive and a list of dependent prims
    '''
    def add_prims(self, prims):
        for prim in prims:
            self.add_prim(prim[0])
            prim[0].prepare(self)


if __name__ == "__main__":
    pass