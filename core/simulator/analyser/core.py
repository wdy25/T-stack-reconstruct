import warnings
from .event import EventType
from core.ir.prims.prim import PrimitiveType
from .components import MatrixUnit, VectorUnit, TransposeUnit, RouterUnit, Memory

class dependency_table_item():
    def __init__(self, event_names, done):
        self.event_names = event_names
        self.done = done

class Core():
    def __init__(self, chip_pos, core_pos, config):
        self.name = f"Core_{core_pos[0]:0>3}_{core_pos[1]:0>3}"
        self.chip_pos = chip_pos
        self.core_pos = core_pos
        self.config = config
        
        self.matrix: MatrixUnit = MatrixUnit("matrix", self.name, 
                                            config["core"]["int8_PE_array_height"] * config["core"]["int8_PE_array_width"],
                                            config["core"]["bf16_PE_array_height"] * config["core"]["bf16_PE_array_width"])
        self.vector: VectorUnit = VectorUnit("vector", self.name, config["core"]["vec_parallelism"])
        self.transpose: TransposeUnit = TransposeUnit("transpose", self.name)
        self.memory: list[Memory] = []
        self.memory.append(Memory("memory_0", self.name, self.config["core"]["L0_memory_bandwidth"]))
        if self.config["core"]["memory_capacity_per_core"] > self.config["core"]["L0_memory_capacity"]:
            self.memory.append(Memory("memory_1", self.name, self.config["core"]["L1_memory_bandwidth"]))
        self.router: RouterUnit = RouterUnit("router", self.name)
        
        self.prims = []
        self.PC = 0
        
        self.running_prims_table = []
        
        self.duration = 0
        
        self.stop = False
        

    def dispatch(self, engine):
        # update "done"
        for running_prim in self.running_prims_table:
            done = 1
            for event_nm in running_prim.event_names:
                if event_nm in engine.all_events:
                    done = 0
            running_prim.done = done
        
        # pop oldeset prims
        i = 0
        while i < len(self.running_prims_table):
            if self.running_prims_table[i].done:
                i += 1
                continue
            else:
                break
        if i > 0:
            for _ in range(i):
                self.running_prims_table.pop(0)
        
        
        if len(self.prims) > 0 and (self.PC >= len(self.prims) or self.stop) and len(self.running_prims_table) == 0 and self.duration == 0:
            self.duration = engine.cur_time_stamp
        elif (not (self.PC >= len(self.prims) or self.stop)) or len(self.running_prims_table) != 0:
            self.duration = 0
        
        
        # dispatch
        while True:
            if self.PC >= len(self.prims):
                break
            if self.stop:
                break
            if len(self.running_prims_table) >= self.config["core"]["max_dependence_window"]:
                break
            
            # check dependencies
            prim = self.prims[self.PC]
            can_run = True
            running_prim_num = len(self.running_prims_table)
            for i in range(len(self.running_prims_table)):
                if i >= len(prim.dependent_prims):
                    if not self.running_prims_table[running_prim_num - i - 1].done:
                        can_run = False
                        break
                    continue
                if prim.dependent_prims[i] and not self.running_prims_table[running_prim_num - i - 1].done:
                    can_run = False
                    break
            if not can_run:
                break
            
            if prim.type == PrimitiveType.CONTROL:
                from core.ir.prims.stop import PrimStop
                from core.ir.prims.jump import PrimJump
                if isinstance(prim, PrimStop) or isinstance(prim, PrimJump):
                    prim.execute(self)
                else:
                    warnings.warn(f"Unsupported control prim {prim.name} in core {self.name}, skip executing.")
                self.running_prims_table.append(dependency_table_item([], 1))
                if self.stop:
                    break
            elif prim.type == PrimitiveType.ROUTER:
                event_list = prim.generate_events(self)
                if event_list[0].router_event.sending == False:
                    if self.router.receiving:
                        can_run = False
                        break
                    if len(event_list) > 1:  # recv and send
                        if self.router.sending:
                            can_run = False
                            break
                else:  # only send
                    if self.router.sending:
                        can_run = False
                        break
                
                flat_event_list = []
                for msg_event in event_list:
                    if msg_event.router_event is not None:
                        flat_event_list.append(msg_event.router_event.full_name)
                    if msg_event.mem_events is not None and len(msg_event.mem_events) > 0 and msg_event.mem_events[0] is not None:
                        flat_event_list.append(msg_event.mem_events[0].full_name)
                    if msg_event.sync_event is not None:
                        flat_event_list.append(msg_event.sync_event.full_name)
                
                self.running_prims_table.append(dependency_table_item(flat_event_list, 0))
                
                self.router.add_events(event_list, engine, self)
                
                self.PC += 1
            
            else:
                # check resource availability
                sending = None
                events = prim.generate_events(self)

                for event in events:
                    match event.event_type:
                        case EventType.MATRIX:
                            if len(self.matrix.running_events) >= 1:
                                can_run = False
                                break
                        case EventType.VECTOR:
                            if len(self.vector.running_events) >= 1:
                                can_run = False
                                break
                        case EventType.TRANSPOSE:
                            if len(self.transpose.running_events) >= 1:
                                can_run = False
                                break                
                if not can_run:
                    break
                
                # dispatching
                event_list = []
                for event in events:
                    event_list.append(event.full_name)
                
                self.running_prims_table.append(dependency_table_item(event_list, 0))
                
                
                for event in events:
                    match event.event_type:
                        case EventType.MATRIX:
                            self.matrix.add_event(event, engine)
                        case EventType.VECTOR:
                            self.vector.add_event(event, engine)
                        case EventType.TRANSPOSE:
                            self.transpose.add_event(event, engine)
                        case EventType.READ:
                            self.memory[event.hierarchy].add_event(event, engine)
                        case EventType.WRITE:
                            self.memory[event.hierarchy].add_event(event, engine)
                        case EventType.INOUT:
                            self.memory[event.hierarchy].add_event(event, engine)
                                
                self.PC += 1
    
    
    def advance(self, engine):        
        self.matrix.advance(engine)
        self.vector.advance(engine)
        self.transpose.advance(engine)
        self.router.advance(engine)
        for memory in self.memory:
            memory.advance(engine)

    
    def add_prim(self, prim, dependent_prims):
        self.prims.append(prim)
        self.prims[-1].dependent_prims = dependent_prims
    
    
    '''
    prims: list of tuples, each tuple contains a primitive and a list of dependent prims
    '''
    def add_prims(self, prims):
        for prim in prims:
            self.add_prim(prim[0], prim[1])