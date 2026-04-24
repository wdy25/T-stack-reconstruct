from functools import lru_cache

from .chip_array import ChipArray

def int32(value):
    return int(value) % 0x7FFFFFFF

class Engine():
    def __init__(self, chip_array: ChipArray, config):
        self.config = config
        self.chip_array = chip_array
        
        self.all_events = {}
        
        self.cur_time_stamp = 0
        self.next_time_stamp = 0
        self.last_time_stamp = 0
        self.stuck_count = 0
        
        self.core_pool = {}
        self.module_pool = {}
        self.event_pool = {}
    
    def run(self):
        self.stuck_count = 0
        self.cur_time_stamp = 0
        self.next_time_stamp = 0
        self.last_time_stamp = 0
        
        while True:
            for core in self.chip_array.cores.values():
                core.router.update_send_recv(self)
            
            for core in self.chip_array.cores.values():
                core.dispatch(self)
            
            for core in self.chip_array.cores.values():
                core.router.sync(self)
            
            for core in self.chip_array.cores.values():
                core.router.resolve_sync(self)
            
            for core in self.chip_array.cores.values():
                for memory in core.memory:
                    memory.resolve_contention(self)
            
            for core in self.chip_array.cores.values():
                for memory in core.memory:
                    memory.update_bounded_events(self)
            
            least_advance_time = float("inf")
            for event in self.all_events.values():
                if event.remaining_time != float('inf') and round(event.remaining_time) < least_advance_time:
                    # print(event.parent, event.name, self.cur_time_stamp + event.remaining_time)
                    least_advance_time = round(event.remaining_time)
            
            if least_advance_time == float("inf"):
                least_advance_time = 0
                self.stuck_count += 1
                if self.stuck_count > 2:
                    break
            
            assert least_advance_time >= 0
            
            self.next_time_stamp = self.cur_time_stamp + least_advance_time
            
            for core in self.chip_array.cores.values():
                core.advance(self)
            
            self.last_time_stamp = self.cur_time_stamp
            self.cur_time_stamp = self.next_time_stamp
        
        print("Simulation finished at time stamp: %d" % self.cur_time_stamp)
        if len(self.all_events) > 0:
            print("Remaining events:")
            for event in self.all_events.keys():
                print(f"{event} not finished")
        
    
    def printUtilizations(self):
        total_computation = 0
        total_energy = 0
        total_data_trans = 0
        for core in self.chip_array.cores.values():
            if core.duration != 0:
                # core.matrix.average_resource_utilization = core.matrix.theoretical_volume / (self.cur_time_stamp * core.matrix.resource_per_cycle)
                # core.vector.average_resource_utilization = core.vector.theoretical_volume / (self.cur_time_stamp * core.vector.resource_per_cycle)
                # core.memory[0].average_resource_utilization = core.memory[0].total_volume / (self.cur_time_stamp * core.memory[0].resource_per_cycle)
                
                core.matrix.average_resource_utilization = core.matrix.total_volume / (core.duration * core.matrix.resource_per_cycle)
                core.matrix.bf16_average_resource_utilization = core.matrix.bf16_total_volumne / (core.duration * core.matrix.bf16_resource) if core.matrix.bf16_resource > 0 else 0
                core.vector.average_resource_utilization = core.vector.theoretical_volume / (core.duration * core.vector.resource_per_cycle)
                core.memory[0].average_resource_utilization = core.memory[0].total_volume / (core.duration * core.memory[0].resource_per_cycle)
                
                print(f"Core {core.name} stop @ {core.duration}\n    matrix: {core.matrix.energy} nJ\n        int: {core.matrix.max_resource_utilization}(max) {core.matrix.average_resource_utilization}(avg)\n        float: {core.matrix.max_bf16_resource_utilization}(max) {core.matrix.bf16_average_resource_utilization}(avg)\n    vector: {core.vector.max_resource_utilization}(max) {core.vector.average_resource_utilization}(avg) {core.vector.energy} nJ\n    memory_0: {core.memory[0].max_resource_utilization}(max) {core.memory[0].average_resource_utilization}(avg) {core.memory[0].energy} nJ\n    router: {core.router.energy} nJ\n    transpose: {core.transpose.energy} nJ")    
                
                # print(core.matrix.theoretical_volume)
                total_computation += core.matrix.total_volume * 2 + core.vector.total_volume
                total_energy += core.matrix.energy + core.vector.energy + core.memory[0].energy + (0 if len(core.memory) == 1 else core.memory[1].energy) + core.router.energy + core.transpose.energy
                total_data_trans += core.router.total_volume
            
        print(f"Total computation: {total_computation}")
        print(f"Total energy: {total_energy} nJ")
        print(f"Total energy efficiency: {total_computation / 1e12 / total_energy / (1e-9)} TOPS/W")
        print(f"Total data transfer: {total_data_trans} bytes")
    
    
    @lru_cache(maxsize=None)
    def getUniqueCoreId(self, core_name):
        name = core_name
        if name in self.core_pool:
            return self.core_pool[name]
        
        core_id = int32(hash(name))
        while core_id in self.core_pool.values():
            core_id = int32(core_id + 1)
        self.core_pool[name] = core_id
        return core_id
    

    @lru_cache(maxsize=None)
    def getUniqueModuleId(self, module_name):
        name = module_name
        if name in self.module_pool:
            return self.module_pool[name]
        
        module_id = int32(hash(name))
        while module_id in self.module_pool.values():
            module_id = int32(module_id + 1)
        self.module_pool[name] = module_id
        return module_id
    
    
    @lru_cache(maxsize=None)
    def getUniqueEventId(self, event_name):
        name = event_name
        if name in self.event_pool:
            return self.event_pool[name]
        
        event_id = int32(hash(name))
        while event_id in self.event_pool.values():
            event_id = int32(event_id + 1)
        self.event_pool[name] = event_id
        return event_id
        
    
    
    def getTrace(self, path=None):        
        from core.utils.trace_protos import trace_pb2 as trace
        t = trace.Trace()
        sequence_id = 3903809
        for core in self.chip_array.cores.values():
            core_id = self.getUniqueCoreId(core.name)
            core_process = t.packet.add()
            core_process.track_descriptor.uuid = core_id
            core_process.track_descriptor.name = core.name
            core_process.track_descriptor.process.pid = core_id
            core_process.track_descriptor.process.process_name = core.name
            
            # matrix
            matrix_id = self.getUniqueModuleId(core.matrix.full_name)
            matrix_thread = t.packet.add()
            matrix_thread.track_descriptor.uuid = matrix_id
            matrix_thread.track_descriptor.parent_uuid = core_id
            matrix_thread.track_descriptor.thread.pid = core_id
            matrix_thread.track_descriptor.thread.tid = matrix_id
            matrix_thread.track_descriptor.thread.thread_name = core.matrix.name
            
            for event in core.matrix.historical_events:
                # print(event.start_time, event.end_time, event.name)
                event_trace = t.packet.add()
                event_trace.timestamp = event.start_time
                event_trace.track_event.name = event.name
                event_trace.track_event.type = trace.TrackEvent.Type.TYPE_SLICE_BEGIN
                event_trace.track_event.track_uuid = matrix_id
                event_trace.trusted_packet_sequence_id = sequence_id

                end_event_trace = t.packet.add()
                end_event_trace.timestamp = event.end_time
                end_event_trace.track_event.type = trace.TrackEvent.Type.TYPE_SLICE_END
                end_event_trace.track_event.track_uuid = matrix_id
                end_event_trace.trusted_packet_sequence_id = sequence_id
            
            # vector
            vector_id = self.getUniqueModuleId(core.vector.full_name)
            vector_thread = t.packet.add()
            vector_thread.track_descriptor.uuid = vector_id
            vector_thread.track_descriptor.parent_uuid = core_id
            vector_thread.track_descriptor.thread.pid = core_id
            vector_thread.track_descriptor.thread.tid = vector_id
            vector_thread.track_descriptor.thread.thread_name = core.vector.name
            
            for event in core.vector.historical_events:
                # print(event.start_time, event.end_time, event.name)
                event_trace = t.packet.add()
                event_trace.timestamp = event.start_time
                event_trace.track_event.name = event.name
                event_trace.track_event.type = trace.TrackEvent.Type.TYPE_SLICE_BEGIN
                event_trace.track_event.track_uuid = vector_id
                event_trace.trusted_packet_sequence_id = sequence_id
                
                end_event_trace = t.packet.add()
                end_event_trace.timestamp = event.end_time
                end_event_trace.track_event.type = trace.TrackEvent.Type.TYPE_SLICE_END
                end_event_trace.track_event.track_uuid = vector_id
                end_event_trace.trusted_packet_sequence_id = sequence_id
            
            # transpose
            transpose_id = self.getUniqueModuleId(core.transpose.full_name)
            transpose_thread = t.packet.add()
            transpose_thread.track_descriptor.uuid = transpose_id
            transpose_thread.track_descriptor.parent_uuid = core_id
            transpose_thread.track_descriptor.thread.pid = core_id
            transpose_thread.track_descriptor.thread.tid = transpose_id
            transpose_thread.track_descriptor.thread.thread_name = core.transpose.name
            
            for event in core.transpose.historical_events:
                # print(event.start_time, event.end_time, event.name)
                event_trace = t.packet.add()
                event_trace.timestamp = event.start_time
                event_trace.track_event.name = event.name
                event_trace.track_event.type = trace.TrackEvent.Type.TYPE_SLICE_BEGIN
                event_trace.track_event.track_uuid = transpose_id
                event_trace.trusted_packet_sequence_id = sequence_id
                
                end_event_trace = t.packet.add()
                end_event_trace.timestamp = event.end_time
                end_event_trace.track_event.type = trace.TrackEvent.Type.TYPE_SLICE_END
                end_event_trace.track_event.track_uuid = transpose_id
                end_event_trace.trusted_packet_sequence_id = sequence_id
                
            # memory
            for memory in core.memory:
                for event in memory.historical_events:
                    event_id = self.getUniqueEventId(event.full_name)
                    track = t.packet.add()
                    track.track_descriptor.uuid = event_id
                    track.track_descriptor.parent_uuid = core_id
                    track.track_descriptor.name = memory.name
                    
                    event_trace = t.packet.add()
                    event_trace.timestamp = event.start_time
                    event_trace.track_event.name = event.name
                    event_trace.track_event.type = trace.TrackEvent.Type.TYPE_SLICE_BEGIN
                    event_trace.track_event.track_uuid = event_id
                    event_trace.trusted_packet_sequence_id = sequence_id
                    
                    end_event_trace = t.packet.add()
                    end_event_trace.timestamp = event.end_time
                    end_event_trace.track_event.type = trace.TrackEvent.Type.TYPE_SLICE_END
                    end_event_trace.track_event.track_uuid = event_id
                    end_event_trace.trusted_packet_sequence_id = sequence_id
            
            # router
            for event in core.router.historical_events:
                event_id = self.getUniqueEventId(event.full_name)
                track = t.packet.add()
                track.track_descriptor.uuid = event_id
                track.track_descriptor.parent_uuid = core_id
                track.track_descriptor.name = core.router.name
                
                if event.end_time > event.start_time:
                    event_trace = t.packet.add()
                    event_trace.timestamp = event.start_time
                    event_trace.track_event.name = event.name
                    event_trace.track_event.type = trace.TrackEvent.Type.TYPE_SLICE_BEGIN
                    event_trace.track_event.track_uuid = event_id
                    event_trace.trusted_packet_sequence_id = sequence_id
                    
                    end_event_trace = t.packet.add()
                    end_event_trace.timestamp = event.end_time
                    end_event_trace.track_event.type = trace.TrackEvent.Type.TYPE_SLICE_END
                    end_event_trace.track_event.track_uuid = event_id
                    end_event_trace.trusted_packet_sequence_id = sequence_id
                else:
                    event_trace = t.packet.add()
                    event_trace.timestamp = event.start_time
                    event_trace.track_event.name = event.name
                    event_trace.track_event.type = trace.TrackEvent.Type.TYPE_INSTANT
                    event_trace.track_event.track_uuid = event_id
                    event_trace.trusted_packet_sequence_id = sequence_id
            
        if path is not None:
            with open(path, "wb") as f:
                f.write(t.SerializeToString())
        
        return t
        
        