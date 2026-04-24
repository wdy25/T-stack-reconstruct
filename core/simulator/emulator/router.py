from __future__ import annotations

from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Core

class RouterType(Enum):
    SENDING = 0
    RECEIVING = 1
    IDLE = 2
    VACANT = 3


class RouterState():
    def __init__(self, router_type: RouterType, tag=None):
        self.router_type = router_type
        self.tag = tag
        if router_type == RouterType.IDLE or router_type == RouterType.VACANT:
            assert tag == None


class RouterRequest():
    def __init__(self, pos_y, pos_x, tag, data):
        self.sender = (pos_y, pos_x)
        self.tag = tag
        self.data = data


class NOC():
    def __init__(self):
        self.core_array: Dict[Tuple[int, int], Core] = {}
        
    def add_core(self, core):
        self.core_array[(core.pos_y, core.pos_x)] = core

    def __getitem__(self, index):
        assert type(index) == tuple, "Index should be a tuple"
        assert len(index) == 2, "Index should be a tuple of two elements"
        return self.core_array[index]