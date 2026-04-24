from basics.config import MAX_CORE_MEM_ADDR_IN_32B, MAX_CORE_MEM_ADDR_IN_8B, MAX_CORE_MEM_ADDR_IN_1B

from copy import deepcopy

class AddrBlocksInMem():
    def __init__(self, data: dict[int, int]=None, addressing="32B"):
        self.addressing = addressing
        self.data = dict[int, int]()
        if data is not None:
            self.data = dict[int, int](data)  # create a new dict instead of using the reference
            for key, value in data.items():
                self.addr_check(key)
                assert isinstance(value, int)
                self.addr_check(key + value - 1)
    
    def addr_check(self, addr):
        assert isinstance(addr, int)
        assert addr >= 0
        if self.addressing == "32B":
            assert addr < MAX_CORE_MEM_ADDR_IN_32B # (1 << config.CORE_MEM_ADDR_WIDTH_IN_32B)
        elif self.addressing == "8B":
            assert addr < MAX_CORE_MEM_ADDR_IN_8B  # (1 << config.CORE_MEM_ADDR_WIDTH_IN_8B)
        elif self.addressing == "1B":
            assert addr < MAX_CORE_MEM_ADDR_IN_1B  # (1 << config.CORE_MEM_ADDR_WIDTH_IN_1B)
        else:
            raise ValueError("Invalid addressing")
    
    def add_data(self, addr, data_length):
        assert isinstance(addr, int)
        assert isinstance(data_length, int)
        self.addr_check(addr)
        self.addr_check(addr + data_length)
        self.data[addr] = data_length
        
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        assert isinstance(key, int)
        assert isinstance(value, int)
        self.addr_check(key)
        self.addr_check(key + value)
        self.data[key] = value
        
    def __len__(self):
        return len(self.data)
