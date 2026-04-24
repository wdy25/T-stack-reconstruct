from enum import Enum
from typing import List, Optional

class EventType(Enum):
    MATRIX = 0
    VECTOR = 1
    TRANSPOSE = 2
    READ = 3
    WRITE = 4
    INOUT = 5
    ROUTER = 6
    SYNC = 7
    NUM = 8


class Event():
    def __init__(self, name, parent):
        # predefined attributes
        self.name = name
        self.volume = 0
        self.energy = 0
        self.parent = parent
        self.max_consume_rate = 0
        self.event_type = None
        
        # runtime attributes
        self.remaining_volume = 0
        self.cur_consume_rate = 0
        self.remaining_time = 0
        
        # final attributes
        self.start_time = None
        self.end_time = None
    
    @property
    def full_name(self):
        return f"{self.parent}_{self.name}"


class ComputeEvent(Event):
    def __init__(self, name, parent, compute_type, computation, theoretical_computation, max_consume_rate, energy, precision="INT8"):
        super().__init__(name, parent)
        assert compute_type in [EventType.MATRIX, EventType.VECTOR, EventType.TRANSPOSE]
        self.event_type = compute_type
        self.volume = computation
        self.theoretical_computation = theoretical_computation
        self.energy = energy
        self.max_consume_rate = max_consume_rate
        self.precision = precision
        
        # runtime
        self.remaining_volume = self.volume
        self.remaining_time = self.volume / max_consume_rate
        self.cur_consume_rate = max_consume_rate


class MemoryEvent(Event):
    def __init__(self, name, parent, memory_type, volume, bounded_events, energy, max_bandwidth, hierarchy=0):
        super().__init__(name, parent)
        assert memory_type in [EventType.READ, EventType.WRITE, EventType.INOUT]
        self.event_type = memory_type
        self.volume = volume
        self.bounded_events = bounded_events
        self.energy = energy
        self.max_consume_rate = max_bandwidth
        self.hierarchy = hierarchy
        
        # runtime
        self.remaining_volume = volume
        self.cur_consume_rate = self.max_consume_rate
        self.remaining_time = self.remaining_volume / self.cur_consume_rate


class CommEvent(Event):
    def __init__(self, name, parent, volume, max_bandwidth, energy, sending, target=None, cell_or_neuron=0, Q=False, CXY=False, mc_y=0, mc_x=0, recv_num=0, tag=None):
        super().__init__(name, parent)
        self.event_type = EventType.ROUTER
        self.volume = volume
        self.max_consume_rate = max_bandwidth
        self.energy = energy
        self.sending = sending
        self.send_target = None if not sending else target
        
        self.Q = Q
        self.cell_or_neuron = cell_or_neuron
        self.CXY = CXY
        self.mc_y = mc_y
        self.mc_x = mc_x
        self.recv_num = recv_num
        self.tag = tag
        self.reset()
        
    def reset(self):
        self.remaining_volume = self.volume
        self.cur_consume_rate = self.max_consume_rate
        self.remaining_time = self.remaining_volume / self.cur_consume_rate
        
        
class SyncEvent(Event):
    def __init__(self, name, parent, position, sync_targets, sending, energy, tag=None, handshake=True, Q=False):
        super().__init__(name, parent)
        self.event_type = EventType.SYNC
        self.energy = energy
        self.sync_targets = sync_targets
        self.synced = False
        self.sending = sending
        self.tag = tag
        self.handshake = handshake
        self.position = position
        self.Q = Q
        
        if self.Q: assert (not self.handshake), "handshake and Q cannot be both true"
        
        self.remaining_time = float("inf")


class MsgEvents():
    def __init__(self, sync_event: Optional[SyncEvent], router_event: Optional[CommEvent], mem_events: Optional[List[MemoryEvent]]):
        self.sync_event = sync_event
        self.router_event = router_event
        self.mem_events = mem_events