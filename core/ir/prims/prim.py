from enum import Enum

from basics.addr_blocks_in_mem import AddrBlocksInMem
from basics.data_block import DataBlock

from myhdl import bin, intbv
from abc import ABC, abstractmethod

class PrimitiveType(Enum):
    CONTROL = 0
    MATRIX = 1
    VECTOR = 2
    ROUTER = 3
    NUM = 4

class Primitive(ABC):
    def __init__(self):
        self.name = ""
        self.PIC = intbv(0, min=0, max=(1<<256))
        self.deps = intbv(255, min=0, max=(1<<8))  # 11111111
        self.data_addr_list = dict[str, AddrBlocksInMem]()  # record the addresses of all data
        
        self.data_blocks = dict[str, DataBlock]()  # record the pre-fed data blocks
        # only the input data should be fed into the primitive
        # till now, the inputs of all the primitives are whole data blocks, at least viewed as whole data blocks
        # so the value of self.data_blocks is only DataBlock, not list[DataBlock]
        self.type = PrimitiveType.NUM
    
    @abstractmethod
    def setPIC(self) -> None:
        pass


if __name__ == "__main__":
    b = Primitive()