from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from core.ir.data import Data

class ControlOp(ABC):

    def __init__(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.attrs = attrs or {}