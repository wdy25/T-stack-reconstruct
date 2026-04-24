from abc import ABC, abstractmethod
from .event import Event

class Module(ABC):
    def __init__(self, name, parent):
        self.module_type = None
        self.name = name
        self.parent = parent
        
        self.running_events: dict[str, Event] = {}
        # {event_name: event}
        
        self.historical_events = []
        
        self.energy = 0
        
        self.resource_per_cycle = 0
        self.max_resource_utilization = 0
        self.average_resource_utilization = 0
        self.total_volume = 0
        self.theoretical_volume = 0
        
    
    @abstractmethod
    def advance(self, engine):
        pass
    
    @abstractmethod
    def add_event(self, event, engine):
        pass
    
    @property
    def full_name(self):
        return f"{self.parent}_{self.name}"