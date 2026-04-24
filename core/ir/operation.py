from copy import deepcopy
from myhdl import bin, intbv
from typing import Any, Dict, Iterable, List, Optional, Tuple
from core.ir.data import Data, DataType
from abc import ABC, abstractmethod

class Operation(ABC):
    """Base class for all operations (including Control).

    Fields:
    - name: human-readable identifier.
    - attrs: op parameters dictionary (kept minimal here for speed/flexibility).

    Extension points:
    - infer(inputs_meta: List[Data]) -> List[(shape, dtype)]: static shape/dtype inference.
    - codegen(inputs_meta: List[Data]) -> str: emit a textual IR/"machine code" for this op.
      Note: This repository emits a simple IR string; integrate with real backends as needed.
    """

    __slots__ = ("name", "attrs")

    def __init__(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.attrs = attrs or {}
        self.primitive = False

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(name={self.name!r})"
    
    @abstractmethod
    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:  # type
        """
        inputs: Dict[int, Data] - mapping from port number to Data node
        returns: List of (shape, dtype) tuples for each output port
        Note that the sequence of the output list corresponds to the output port id.
        """
        pass
            
    
    # @abstractmethod
    # def paragen(self, inputs: List[Data]):
    #     pass
    
    #     # Optional: override in subclasses that support compilation/codegen
    # @abstractmethod
    # def codegen(self, inputs: List[Data]):  # type: ignore[name-defined]
    #     pass