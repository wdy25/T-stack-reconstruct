from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from core.ir.graph import Graph
from core.ir.data import Data, DataType, ViewData, create_reshape_view
from core.ir.operation import Operation
from core.ir.operations.layernorm import LayerNorm
from core.ir.operations.mat_mul import MatMul
from core.ir.operations.lut import LUT


class FeedForwardLayer(Operation):
    """Vision Transformer feed-forward block composed from existing ops: LayerNorm -> Linear1(MatMul) -> GELU(LUT) -> Linear2(MatMul).

    Ports:
        inputs:
            - 0: Input activations (2D tensor (batch_size, embed_dim), BF16)
            - 1: First linear weight matrix (2D tensor: (embed_dim, hidden_dim), BF16)
            - 2: First linear bias vector (1D or 2D tensor, (hidden_dim,) or (1, hidden_dim), BF16)
            - 3: Second linear weight matrix (2D tensor: (hidden_dim, embed_dim), BF16)
            - 4: Second linear bias vector (1D or 2D tensor, (embed_dim,) or (1, embed_dim), BF16)
            - 5: GELU lookup table (2D tensor with 65536 BF16 entries or 256 INT8 entries)
        outputs:
            - 0: Output activations matching the input shape and dtype.

    Required attrs:
        - input_shape (Tuple[int, ...]): Expected shape of the input tensor.
            - embed_dim (int): Size of the model embedding dimension (input last dimension).
        - hidden_dim (int): Width of the intermediate linear layer.
        - epsilon (float): LayerNorm epsilon for numerical stability.

    Optional attrs:
        - normalized_shape (Tuple[int, ...]): LayerNorm normalized shape. Defaults to (embed_dim,) and currently must be (embed_dim,).

    Notes:
        - Dropout layers from the PyTorch reference are omitted by design.
        - LayerNorm uses the existing composite operation with shared input for both ports.
    """

    def __init__(self, name: str, attrs: Optional[Dict[str, object]] = None) -> None:
        super().__init__(name, attrs)
        attrs = self.attrs
        required_attrs = ["input_shape", "hidden_dim", "epsilon", ]
        for attr in required_attrs:
            if attr not in attrs:
                raise ValueError(f"Missing required attribute '{attr}' for FeedForwardLayer.")

        hidden_dim = int(attrs["hidden_dim"])
        if hidden_dim <= 0:
            raise ValueError("'hidden_dim' must be a positive integer.")
        epsilon = float(attrs["epsilon"])
        if epsilon <= 0.0:
            raise ValueError("'epsilon' must be positive.")

        input_shape = attrs["input_shape"]
        if not isinstance(input_shape, tuple):
            raise ValueError("'input_shape' must be a tuple of positive integers.")
        if any(dim <= 0 for dim in input_shape):
            raise ValueError("'input_shape' values must all be positive integers.")
        if len(input_shape) != 2:
            raise ValueError("'input_shape' must describe a 2D tensor (batch_size, embed_dim).")

        self.primitive = False  # mark as non-primitive op

        self._flat_batch: int = math.prod(input_shape[:-1])
        self._embed_dim: int = input_shape[-1]
        self._input_shape: Tuple[int, ...] = input_shape
        self._hidden_dim: int = hidden_dim
        self._epsilon: float = epsilon
        self._normalized_shape: Tuple[int, ...] = (self._embed_dim,)

        # ensure stored normalized_shape back into attrs for downstream consumers
        self.attrs["normalized_shape"] = self._normalized_shape

    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        for port in range(6):
            if port not in inputs:
                raise ValueError(f"FeedForwardLayer expects input on port {port}.")

        input_data = inputs[0]
        if input_data.shape is None:
            raise ValueError("Input tensor must have a defined shape for FeedForwardLayer.")
        if len(input_data.shape) != 2:
            raise ValueError("Input tensor must be a 2D tensor (batch_size, embed_dim) for FeedForwardLayer.")
        if input_data.dtype != DataType.BF16:
            raise ValueError("FeedForwardLayer currently supports BF16 inputs only.")

        input_shape = input_data.shape
        if input_shape != self._input_shape:
            raise ValueError(
                f"Input tensor shape {input_shape} does not match configured input_shape {self._input_shape}."
            )

        embed_dim = self._embed_dim
        hidden_dim = self._hidden_dim

        def _require_matrix(data: Data, rows: int, cols: int, name: str) -> None:
            if data.shape is None or len(data.shape) != 2:
                raise ValueError(f"{name} must be a 2D tensor, got {data.shape}.")
            if data.shape[0] != rows or data.shape[1] != cols:
                raise ValueError(f"{name} shape must be ({rows}, {cols}), got {data.shape}.")

        def _require_bias(data: Data, size: int, name: str) -> None:
            if data.shape is None:
                raise ValueError(f"{name} must have defined shape.")
            if len(data.shape) == 1:
                if data.shape[0] != size:
                    raise ValueError(f"{name} 1D shape must be ({size},), got {data.shape}.")
            elif len(data.shape) == 2:
                if data.shape != (1, size):
                    raise ValueError(f"{name} 2D shape must be (1, {size}), got {data.shape}.")
            else:
                raise ValueError(f"{name} dim must be 1 or 2, got {len(data.shape)}.")

        _require_matrix(inputs[1], embed_dim, hidden_dim, "First linear weight")
        _require_bias(inputs[2], hidden_dim, "First linear bias")
        _require_matrix(inputs[3], hidden_dim, embed_dim, "Second linear weight")
        _require_bias(inputs[4], embed_dim, "Second linear bias")

        lut_data = inputs[5]
        if lut_data.dtype not in (DataType.BF16, DataType.INT8):
            raise ValueError("GELU LUT tensor must use BF16 or INT8 dtype.")
        if lut_data.shape is None or len(lut_data.shape) != 2:
            raise ValueError("GELU LUT tensor must be 2D.")
        lut_shape = lut_data.shape
        total_lut_entries = lut_shape[0] * lut_shape[1]
        if total_lut_entries not in (65536, 256):
            raise ValueError("GELU LUT tensor must have 65536 entries for BF16 input or 256 entries for INT8 input.")

        return [(input_shape, input_data.dtype)]

    def to_prim(self):
        """Lower the high-level feed-forward block into primitives-ready subgraph.

        The lowering follows these steps:
            1. Forward the 2D input through an identity reshape view to fan out ports.
            2. Apply LayerNorm with provided epsilon over the feature dimension.
            3. Execute the first linear projection via MatMul + bias.
            4. Apply GELU using the LUT primitive with BF16 (or INT8) lookup table.
            5. Execute the second linear projection via MatMul + bias.
        """
        subgraph = Graph()

        layernorm_attrs = {
            "normalized_shape": self._normalized_shape,
            "epsilon": self._epsilon,
            "elementwise_affine": False,
        }
        layernorm_op = LayerNorm(name=f"{self.name}_layernorm", attrs=layernorm_attrs)
        subgraph.add_node(layernorm_op)

        layernorm_out = Data(name=f"{self.name}_layernorm_out")
        subgraph.add_node(layernorm_out)
        subgraph.connect(layernorm_op.name, layernorm_out.name, 0, 0)

        fc1 = MatMul(
            name=f"{self.name}_fc1",
            attrs={
                "in_channels": self._embed_dim,
                "out_channels": self._hidden_dim,
                "batch_size": self._flat_batch,
            },
        )
        subgraph.add_node(fc1)
        subgraph.connect(layernorm_out.name, fc1.name, 0, 0)

        fc1_out = Data(name=f"{self.name}_fc1_out")
        subgraph.add_node(fc1_out)
        subgraph.connect(fc1.name, fc1_out.name, 0, 0)

        gelu = LUT(
            name=f"{self.name}_gelu_lut",
            attrs={"input_dtype": DataType.BF16, "output_dtype": DataType.BF16},
        )
        subgraph.add_node(gelu)
        subgraph.connect(fc1_out.name, gelu.name, 0, 0)

        gelu_out = Data(name=f"{self.name}_gelu_lut_out")
        subgraph.add_node(gelu_out)
        subgraph.connect(gelu.name, gelu_out.name, 0, 0)

        fc2 = MatMul(
            name=f"{self.name}_fc2",
            attrs={
                "in_channels": self._hidden_dim,
                "out_channels": self._embed_dim,
                "batch_size": self._flat_batch,
            },
        )
        subgraph.add_node(fc2)
        subgraph.connect(gelu_out.name, fc2.name, 0, 0)

        input_mapping = {
            0: [(layernorm_op.name, 0)],
            1: [(fc1.name, 1)],
            2: [(fc1.name, 2)],
            3: [(fc2.name, 1)],
            4: [(fc2.name, 2)],
            5: [(gelu.name, 1)],
        }

        output_mapping = {0: (fc2.name, 0)}

        return {
            "subgraph": subgraph,
            "input_mapping": input_mapping,
            "output_mapping": output_mapping,
        }
