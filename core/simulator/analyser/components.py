from __future__ import annotations

from enum import Enum
import warnings
from typing import List, TYPE_CHECKING
import numpy as np
from .module import Module
from .event import EventType, CommEvent, MemoryEvent, MsgEvents

if TYPE_CHECKING:
    from .engine import Engine
    from .core import Core

class ComponentType(Enum):
    MEMORY = 0
    MATRIX = 1
    VECTOR = 2
    TRANSPOSE = 3
    ROUTER = 4
    NUM = 5


class Memory(Module):
    def __init__(self, name, parent, bandwidth):
        super().__init__(name, parent)
        self.bandwidth = bandwidth
        self.resource_per_cycle = bandwidth
        self.module_type = ComponentType.MEMORY
        
    
    def resolve_contention(self, engine: Engine):
        total_bandwidth_request = 0
        for event in self.running_events.values():
            total_bandwidth_request += event.max_consume_rate
        
        if total_bandwidth_request <= self.bandwidth:
            if total_bandwidth_request > self.max_resource_utilization * self.bandwidth:
                self.max_resource_utilization = total_bandwidth_request / self.bandwidth
            for event in self.running_events.values():
                event.cur_consume_rate = event.max_consume_rate
                event.remaining_time = event.remaining_volume / event.max_consume_rate
        else:
            self.max_resource_utilization = 1
            for event in self.running_events.values():
                event.cur_consume_rate = event.max_consume_rate * self.bandwidth / total_bandwidth_request
                event.remaining_time = event.remaining_volume / event.cur_consume_rate
    
    
    def update_bounded_events(self, engine: Engine):
        for event in self.running_events.values():
            min_consume_ratio = event.cur_consume_rate / event.max_consume_rate
            for bounded_event_nm in event.bounded_events:
                bd_event = engine.all_events[bounded_event_nm]
                if bd_event.cur_consume_rate / bd_event.max_consume_rate < min_consume_ratio:
                    # print(f"Warning: Bounded event {bd_event.full_name} has lower consume ratio than memory event {event.full_name}")
                    min_consume_ratio = bd_event.cur_consume_rate / bd_event.max_consume_rate
            for bounded_event_nm in event.bounded_events:
                bd_event = engine.all_events[bounded_event_nm]
                bd_event.cur_consume_rate = bd_event.max_consume_rate * min_consume_ratio
                bd_event.remaining_time = bd_event.remaining_volume / bd_event.cur_consume_rate
            
            # update the memory event itself
            event.cur_consume_rate = event.max_consume_rate * min_consume_ratio
            event.remaining_time = event.remaining_volume / event.cur_consume_rate                    
    
    def advance(self, engine: Engine):
        advance_time = engine.next_time_stamp - engine.cur_time_stamp
        events_to_remove = []
        for nm, event in self.running_events.items():
            assert round(event.remaining_time) >= advance_time
            if round(event.remaining_time) == advance_time:
                events_to_remove.append(nm)
            else:
                event.remaining_volume -= event.cur_consume_rate * advance_time
                event.cur_consume_rate = event.max_consume_rate  # reset consume rate
                event.remaining_time = event.remaining_volume / event.cur_consume_rate
        
        for nm in events_to_remove:
            self.energy += self.running_events[nm].energy
            self.total_volume += self.running_events[nm].volume
            
            self.running_events[nm].end_time = engine.next_time_stamp
            self.historical_events.append(self.running_events[nm])
            self.running_events.pop(nm)
            engine.all_events.pop(nm)
    
    
    def add_event(self, event, engine: Engine):
        assert event.event_type in [EventType.READ, EventType.WRITE, EventType.INOUT]
        event.start_time = engine.cur_time_stamp
        
        self.running_events[event.full_name] = event
        
        assert event.full_name not in engine.all_events
        engine.all_events[event.full_name] = event
        

class MatrixUnit(Module):
    def __init__(self, name, parent, int8_PE_num, bf16_PE_num):
        super().__init__(name, parent)
        self.int8_resource = int8_PE_num
        self.bf16_resource = bf16_PE_num
        self.resource_per_cycle = int8_PE_num  # default
        self.bf16_theoretical_volume = 0
        self.bf16_total_volumne = 0
        self.module_type = ComponentType.MATRIX
        self.max_bf16_resource_utilization = 0
    
    def advance(self, engine):
        advance_time = engine.next_time_stamp - engine.cur_time_stamp
        events_to_remove = []
        for nm, event in self.running_events.items():
            resource = self.int8_resource if event.precision == "INT8" else self.bf16_resource
            if event.precision == "INT8":
                if event.cur_consume_rate / resource > self.max_resource_utilization:
                    self.max_resource_utilization = event.cur_consume_rate / resource
            if event.precision == "BF16":
                if event.cur_consume_rate / self.bf16_resource > self.max_bf16_resource_utilization:
                    self.max_bf16_resource_utilization = event.cur_consume_rate / self.bf16_resource

            assert round(event.remaining_time) >= advance_time
            if round(event.remaining_time) == advance_time:
                events_to_remove.append(nm)
            else:
                event.remaining_volume -= event.cur_consume_rate * advance_time
                event.cur_consume_rate = event.max_consume_rate  # reset consume rate
                event.remaining_time = event.remaining_volume / event.cur_consume_rate
        
        for nm in events_to_remove:
            self.energy += self.running_events[nm].energy
            self.total_volume += self.running_events[nm].volume if self.running_events[nm].precision == "INT8" else 0
            self.bf16_total_volumne += self.running_events[nm].volume if self.running_events[nm].precision == "BF16" else 0
            self.theoretical_volume += self.running_events[nm].theoretical_computation if self.running_events[nm].precision == "INT8" else 0
            self.bf16_theoretical_volume += self.running_events[nm].theoretical_computation if self.running_events[nm].precision == "BF16" else 0
            
            # print(nm, self.running_events[nm].theoretical_computation / (engine.next_time_stamp - self.running_events[nm].start_time) / self.resource_per_cycle)
            
            self.running_events[nm].end_time = engine.next_time_stamp
            self.historical_events.append(self.running_events[nm])
            self.running_events.pop(nm)
            engine.all_events.pop(nm)
    
    
    def add_event(self, event, engine):
        # do not allow overlapping matrix events
        assert len(self.running_events) == 0
        
        assert event.event_type == EventType.MATRIX
        event.start_time = engine.cur_time_stamp
        
        self.running_events[event.full_name] = event
        
        assert event.full_name not in engine.all_events
        engine.all_events[event.full_name] = event
        

class VectorUnit(Module):
    def __init__(self, name, parent, PE_num):
        super().__init__(name, parent)
        self.resource_per_cycle = PE_num
        self.module_type = ComponentType.VECTOR
    
    def advance(self, engine):
        advance_time = engine.next_time_stamp - engine.cur_time_stamp
        events_to_remove = []
        for nm, event in self.running_events.items():
            if event.cur_consume_rate / self.resource_per_cycle > self.max_resource_utilization:
                self.max_resource_utilization = event.cur_consume_rate / self.resource_per_cycle
            assert round(event.remaining_time) >= advance_time
            if round(event.remaining_time) == advance_time:
                events_to_remove.append(nm)
            else:
                event.remaining_volume -= event.cur_consume_rate * advance_time
                event.cur_consume_rate = event.max_consume_rate  # reset consume rate
                event.remaining_time = event.remaining_volume / event.cur_consume_rate
        
        for nm in events_to_remove:
            self.energy += self.running_events[nm].energy
            self.total_volume += self.running_events[nm].volume
            self.theoretical_volume += self.running_events[nm].theoretical_computation
            
            self.running_events[nm].end_time = engine.next_time_stamp
            self.historical_events.append(self.running_events[nm])
            self.running_events.pop(nm)
            engine.all_events.pop(nm)
    
    
    def add_event(self, event, engine):
        # do not allow overlapping vector events
        assert len(self.running_events) == 0
        
        assert event.event_type == EventType.VECTOR
        event.start_time = engine.cur_time_stamp
        
        self.running_events[event.full_name] = event
        
        assert event.full_name not in engine.all_events
        engine.all_events[event.full_name] = event
    
    
class TransposeUnit(Module):
    def __init__(self, name, parent):
        super().__init__(name, parent)
        self.module_type = ComponentType.TRANSPOSE
    
    def advance(self, engine):
        advance_time = engine.next_time_stamp - engine.cur_time_stamp
        events_to_remove = []
        for nm, event in self.running_events.items():
            assert round(event.remaining_time) >= advance_time
            if round(event.remaining_time) == advance_time:
                events_to_remove.append(nm)
            else:
                event.remaining_volume -= event.cur_consume_rate * advance_time
                event.cur_consume_rate = event.max_consume_rate  # reset consume rate
                event.remaining_time = event.remaining_volume / event.cur_consume_rate
        
        for nm in events_to_remove:
            self.energy += self.running_events[nm].energy
            self.total_volume += self.running_events[nm].volume
            
            self.running_events[nm].end_time = engine.next_time_stamp
            self.historical_events.append(self.running_events[nm])
            self.running_events.pop(nm)
            engine.all_events.pop(nm)
    
    
    def add_event(self, event, engine):
        # do not allow overlapping transpose events
        assert len(self.running_events) == 0
        
        assert event.event_type == EventType.TRANSPOSE
        event.start_time = engine.cur_time_stamp
        
        self.running_events[event.full_name] = event
        
        assert event.full_name not in engine.all_events
        engine.all_events[event.full_name] = event
        

class RouterUnit(Module):
    def __init__(self, name, parent):
        super().__init__(name, parent)
        self.module_type = ComponentType.ROUTER
        
        self.sending = False
        self.receiving = False
        
        self.running_recv_event: str = None
        self.running_write_event: List[str] = []
        self.pending_messages: List[MsgEvents] = None
        self.pending_send_event: CommEvent = None
        self.pending_read_event: MemoryEvent = None
        
        self.tag = None
        self.remaining_recv_num = 0
        self.CXY = False
        self.mc_y = None
        self.mc_x = None
    
    
    def add_recv_event(self, event, engine):
        assert event.event_type == EventType.ROUTER
        event.start_time = engine.cur_time_stamp
        self.running_events[event.full_name] = event
        assert event.full_name not in engine.all_events
        engine.all_events[event.full_name] = event
        
        self.receiving = True
        self.tag = event.tag
        self.remaining_recv_num = event.recv_num
        self.CXY = event.CXY
        self.mc_y = event.mc_y
        self.mc_x = event.mc_x
    
    
    def add_event(self, event, engine):
        assert event.event_type == EventType.ROUTER or event.event_type == EventType.SYNC
        event.start_time = engine.cur_time_stamp
        self.running_events[event.full_name] = event
        assert event.full_name not in engine.all_events
        engine.all_events[event.full_name] = event
        
    
    def add_events(self, event_list: List[MsgEvents], engine: Engine, core: Core):
        self.pending_messages = event_list
        if event_list[0].router_event.sending == False:
            assert self.receiving == False, "Router is already receiving"

            self.running_recv_event = event_list[0].router_event.full_name
            self.add_recv_event(event_list[0].router_event, engine)
            
            self.pending_messages.pop(0)
        
        if self.pending_messages and self.pending_messages[0].router_event.sending == True:
            assert self.sending == False, "Router is already sending"
            self.sending = True
            
            self.pending_send_event = self.pending_messages[0].router_event
            self.pending_read_event = self.pending_messages[0].mem_events[0]
            self.add_event(self.pending_messages[0].sync_event, engine)
            self.pending_messages.pop(0)  
    
    
    def sync(self, engine: Engine):        
        for event in self.running_events.values():
            if event.event_type == EventType.SYNC and event.synced == False:  # all sync events that are not synced
                assert event.sending == True, "Only support sending side to drive the sync for now"
                assert len(event.sync_targets) > 0, "Sync event should have sync targets"
                assert len(event.sync_targets) == 1, "Only support 1-to-1 sync for now"
                
                for target in event.sync_targets:
                    target_core = engine.chip_array[target]
                    target_router = target_core.router
                    if target_router.receiving and (target_router.tag == event.tag or not event.handshake):
                        if target_router.CXY and event.Q:
                            mc_target = (target[0] + target_router.mc_y, target[1] + target_router.mc_x)
                            mc_target_core = engine.chip_array[mc_target]
                            mc_target_router = mc_target_core.router
                            if mc_target_router.receiving:
                                event.synced = True
                                event.end_time = engine.cur_time_stamp
                            else:
                                warnings.warn(f"multicast target {mc_target} is not receiving, send may fail")
                        else:
                            event.synced = True
                    else:
                        if not event.handshake:
                            warnings.warn(f"Sync event {event.full_name} handshake is false but target {target} is not ready, sync may fail")

    
    def resolve_sync(self, engine: Engine):
        events_to_remove = []
        to_send = False
        to_recv = False
        self_position = None
        
        for event in self.running_events.values():
            if event.event_type == EventType.SYNC and event.synced == True:
                events_to_remove.append(event.full_name)
                if event.sending:
                    to_send = True
                else:
                    assert False, "Only support sending side to drive the sync for now"
                self_position = event.position
        
        for nm in events_to_remove:
            self.running_events[nm].end_time = engine.cur_time_stamp  
            self.historical_events.append(self.running_events[nm])
            self.running_events.pop(nm)
            engine.all_events.pop(nm)
        
        
        if self_position is not None:
            core = engine.chip_array[self_position]
            if to_send:
                self.add_event(self.pending_send_event, engine)
                core.memory[0].add_event(self.pending_read_event, engine)
                
                target_core = engine.chip_array[self.pending_send_event.send_target]
                assert target_core.router.receiving, f"Target core {self.pending_send_event.send_target} is not receiving"
                
                def update_recv_events(recv_event: CommEvent, write_event: MemoryEvent):
                    # update bounded events
                    write_event.bounded_events.append(self.pending_read_event.full_name)
                    write_event.bounded_events.append(self.pending_send_event.full_name)
                    
                    self.pending_read_event.bounded_events.append(write_event.full_name)
                    # self.pending_read_event.bounded_events.append(recv_event.full_name)
                    
                    # update the recv event
                    recv_event.volume += self.pending_send_event.volume
                    
                recv_event = engine.all_events[target_core.router.running_recv_event]
                
                write_times = self.pending_send_event.volume / (8 if self.pending_send_event.cell_or_neuron == 0 else 1) * 32 / core.config['core']['memory_width']
                write_event = MemoryEvent(
                    name=f"Recv_write_for_{self.pending_send_event.full_name}",
                    parent=target_core.memory[0].full_name,
                    memory_type=EventType.WRITE,
                    volume=write_times * core.config['core']['memory_width'],
                    bounded_events=[],
                    energy=write_times * core.config["core"]["L0_memory_write_energy"],
                    max_bandwidth=core.config['core']['memory_width']
                )
                
                target_core.memory[0].add_event(write_event, engine)
                target_core.router.running_write_event.append(write_event.full_name)
                update_recv_events(recv_event, write_event)
                
                if self.pending_send_event.Q and target_core.router.CXY:
                    # if it's a multicast, also update the corresponding events in the other target
                    mc_target = (self.pending_send_event.send_target[0] + target_core.router.mc_y, self.pending_send_event.send_target[1] + target_core.router.mc_x)
                    mc_target_core = engine.chip_array[mc_target]
                    assert mc_target_core.router.receiving, f"Multicast target core {mc_target} is not receiving"
                    
                    mc_recv_event = engine.all_events[mc_target_core.router.running_recv_event]
                    
                    mc_write_times = self.pending_send_event.volume / (8 if self.pending_send_event.cell_or_neuron == 0 else 1) * 32 / mc_target_core.config['core']['memory_width']
                    mc_write_event = MemoryEvent(
                        name=f"MC_Recv_write_for_{self.pending_send_event.full_name}",
                        parent=mc_target_core.memory[0].full_name,
                        memory_type=EventType.WRITE,
                        volume=mc_write_times * mc_target_core.config['core']['memory_width'],
                        bounded_events=[],
                        energy=mc_write_times * mc_target_core.config["core"]["L0_memory_write_energy"],
                        max_bandwidth=mc_target_core.config['core']['memory_width']
                    )
                    mc_target_core.memory[0].add_event(mc_write_event, engine)
                    mc_target_core.router.running_write_event.append(mc_write_event.full_name)
                    update_recv_events(mc_recv_event, mc_write_event)
                
                self.pending_read_event = None
                self.pending_send_event = None
                # self.pending_messages.pop(0)
    
    def update_send_recv(self, engine: Engine):
        if self.receiving:
            for write_event_nm in self.running_write_event:
                if write_event_nm not in engine.all_events:
                    self.running_write_event.remove(write_event_nm)
            
            if self.remaining_recv_num == 0:
                self.receiving = False
                assert all(write_event not in engine.all_events for write_event in self.running_write_event), "Write event should have been finished when the write is done"
                self.running_write_event = []
                
                nm = self.running_recv_event
                self.running_events[nm].end_time = engine.cur_time_stamp
                self.historical_events.append(self.running_events[nm])
                self.running_events.pop(nm)
                engine.all_events.pop(nm)
                
                self.running_recv_event = None
                
        if self.sending:
            for event in self.running_events.values():
                if event.event_type == EventType.ROUTER and event.sending == True:
                    return
                if event.event_type == EventType.SYNC and event.sending == True:
                    return
            if self.pending_messages and self.pending_messages[0].router_event.sending == True:
                self.pending_send_event = self.pending_messages[0].router_event
                self.pending_read_event = self.pending_messages[0].mem_events[0]
                self.add_event(self.pending_messages[0].sync_event, engine)
                self.pending_messages.pop(0)
    
    def advance(self, engine):
        advance_time = engine.next_time_stamp - engine.cur_time_stamp
        events_to_remove = []
        
        def solve_recv_event(target_core: Core):
            target_router = target_core.router
            target_router.remaining_recv_num -= 1         
        
        for nm, event in self.running_events.items():
            if event.event_type == EventType.ROUTER:
                assert event.remaining_time == float('inf') or round(event.remaining_time) >= advance_time
                if event.sending:
                    if round(event.remaining_time) == advance_time:
                        events_to_remove.append(nm)
                        
                        target = event.send_target
                        target_core = engine.chip_array[target]
                        solve_recv_event(target_core)
                        
                        if event.Q and target_core.router.CXY:
                            mc_target = (target[0] + target_core.router.mc_y, target[1] + target_core.router.mc_x)
                            mc_target_core = engine.chip_array[mc_target]
                            solve_recv_event(mc_target_core)
                        
                    else:
                        event.remaining_volume -= event.cur_consume_rate * advance_time
                        event.cur_consume_rate = event.max_consume_rate  # reset consume rate
                        event.remaining_time = event.remaining_volume / event.cur_consume_rate
                else:
                    # recv event is driven by the send event, so it does not advance by itself
                    pass
        
        for nm in events_to_remove:
            self.energy += self.running_events[nm].energy
            self.total_volume += self.running_events[nm].volume
            
            self.running_events[nm].end_time = engine.next_time_stamp
            self.historical_events.append(self.running_events[nm])
            self.running_events.pop(nm)
            engine.all_events.pop(nm)