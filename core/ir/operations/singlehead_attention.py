import math
from typing import Dict, List, Optional, Tuple

import torch

from core.ir.graph import Graph
from core.ir.data import Data, DataType, create_reshape_view
from core.ir.operation import Operation
from core.ir.operations.mat_mul import MatMul
from core.ir.operations.multiply import Multiply
from core.ir.operations.softmax import Softmax


class SingleheadAttention(Operation):
    """Single-head self-attention operation.

    Implements the core attention mechanism for a single head:
        1. Project input to Q/K/V using three MatMul ops with dedicated weights
        2. Compute attention scores: Q @ K^T
        3. Scale by 1/sqrt(head_dim)
        4. Apply softmax to get attention weights
        5. Compute output: attention_weights @ V

    This operation is designed to be used as a building block for multi-head
    attention, where each head has its own set of projection weights.

    Required Attributes:
        seq_len: Sequence length (positive integer)
        head_dim: Dimension of the attention head (positive integer)
        
    Optional Attributes:
        scaler: Scaling factor for attention scores (default: 1.0 / math.sqrt(head_dim))
        use_separate_projections: If True, uses separate Q/K/V weights (default: True)
        use_bias: Whether to use bias in projections (default: True)
    """

    def __init__(self, name: str, attrs: Optional[Dict[str, object]] = None) -> None:
        super().__init__(name, attrs)
        
        required_attrs = ["seq_len", "head_dim"]
        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for SingleheadAttention.")

        if not isinstance(self.attrs["seq_len"], int) or self.attrs["seq_len"] <= 0:
            raise ValueError("'seq_len' must be a positive integer.")
        
        if not isinstance(self.attrs["head_dim"], int) or self.attrs["head_dim"] <= 0:
            raise ValueError("'head_dim' must be a positive integer.")
        
        # Set default scaler to 1/sqrt(head_dim)
        head_dim = self.attrs["head_dim"]
        self.attrs.setdefault("scaler", 1.0 / math.sqrt(head_dim))
        
        # Whether this head does its own Q/K/V projections or receives pre-projected inputs
        self.attrs.setdefault("use_separate_projections", True)
        self.attrs.setdefault("use_bias", True)
        
        # Weight names for Q/K/V projections (if use_separate_projections=True)
        self.attrs.setdefault("q_weight_name", f"{name}_q_weight")
        self.attrs.setdefault("k_weight_name", f"{name}_k_weight")
        self.attrs.setdefault("v_weight_name", f"{name}_v_weight")
        self.attrs.setdefault("q_bias_name", f"{name}_q_bias")
        self.attrs.setdefault("k_bias_name", f"{name}_k_bias")
        self.attrs.setdefault("v_bias_name", f"{name}_v_bias")

        self.primitive = False

    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        """Infer output shape and dtype.
        
        Args:
            inputs: Dictionary mapping port numbers to Data nodes
                    Port 0: Input tokens (seq_len, input_dim) or (seq_len, head_dim) if pre-projected
                    Port 1-3: Optional pre-projected Q/K/V if use_separate_projections=False
        
        Returns:
            List with single tuple of (output_shape, output_dtype)
        """
        if 0 not in inputs:
            raise ValueError("SingleheadAttention expects input on port 0.")

        input_data = inputs[0]
        if input_data.shape is None or len(input_data.shape) != 2:
            raise ValueError("Input must be 2D: (seq_len, feature_dim).")

        seq_len, input_dim = input_data.shape
        head_dim = int(self.attrs["head_dim"])
        expected_seq_len = int(self.attrs["seq_len"])

        # Validate that input sequence length matches expected
        if seq_len != expected_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} does not match expected seq_len {expected_seq_len}."
            )

        if input_data.dtype != DataType.BF16:
            raise ValueError("SingleheadAttention currently supports only BF16 inputs.")

        # Output has same shape as input if using projections,
        # or (seq_len, head_dim) if using pre-projected inputs
        if self.attrs.get("use_separate_projections", True):
            output_shape = (seq_len, head_dim)
        else:
            output_shape = input_data.shape

        return [(output_shape, input_data.dtype)]

    def _build_weight(self, name: str, shape: Tuple[int, ...]) -> Data:
        """Create a BF16 weight tensor."""
        tensor = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        return Data(name=name, shape=shape, dtype=DataType.BF16, payload=tensor)

    def _build_bias(self, name: str, length: int) -> Data:
        """Create a zero BF16 bias vector."""
        tensor = torch.zeros(length, dtype=torch.bfloat16)
        return Data(name=name, shape=(length,), dtype=DataType.BF16, payload=tensor)

    def to_prim(self):
        """Convert to primitive operations subgraph."""
        seq_len = int(self.attrs["seq_len"])
        head_dim = int(self.attrs["head_dim"])
        scaler = float(self.attrs["scaler"])
        use_projections = bool(self.attrs.get("use_separate_projections", True))
        use_bias = bool(self.attrs.get("use_bias", True))

        subgraph = Graph()

        # Input buffer for external connection
        input_buffer = Data(name=f"{self.name}_input_buffer", shape=None, dtype=None)
        subgraph.add_node(input_buffer)

        if use_projections:
            # Create Q/K/V projection weights
            input_dim = head_dim  # Assume input is already correct dimension
            q_weight = self._build_weight(self.attrs["q_weight_name"], (input_dim, head_dim))
            k_weight = self._build_weight(self.attrs["k_weight_name"], (input_dim, head_dim))
            v_weight = self._build_weight(self.attrs["v_weight_name"], (input_dim, head_dim))

            subgraph.add_node(q_weight)
            subgraph.add_node(k_weight)
            subgraph.add_node(v_weight)

            q_bias = self._build_bias(self.attrs["q_bias_name"], head_dim) if use_bias else None
            k_bias = self._build_bias(self.attrs["k_bias_name"], head_dim) if use_bias else None
            v_bias = self._build_bias(self.attrs["v_bias_name"], head_dim) if use_bias else None

            for bias in [q_bias, k_bias, v_bias]:
                if bias is not None:
                    subgraph.add_node(bias)

            # Q projection
            q_matmul = MatMul(
                name=f"{self.name}_q_proj",
                attrs={
                    "in_channels": input_dim,
                    "out_channels": head_dim,
                    "batch_size": seq_len,
                    "input_dtype": DataType.BF16,
                    "weight_dtype": DataType.BF16,
                },
            )
            subgraph.add_node(q_matmul)
            subgraph.connect(input_buffer.name, q_matmul.name, 0, 0)
            subgraph.connect(q_weight.name, q_matmul.name, 0, 1)
            if q_bias is not None:
                subgraph.connect(q_bias.name, q_matmul.name, 0, 2)

            q_data = Data(name=f"{self.name}_q", shape=None, dtype=None)
            subgraph.add_node(q_data)
            subgraph.connect(q_matmul.name, q_data.name, 0, 0)

            # K projection
            k_matmul = MatMul(
                name=f"{self.name}_k_proj",
                attrs={
                    "in_channels": input_dim,
                    "out_channels": head_dim,
                    "batch_size": seq_len,
                    "input_dtype": DataType.BF16,
                    "weight_dtype": DataType.BF16,
                },
            )
            subgraph.add_node(k_matmul)
            subgraph.connect(input_buffer.name, k_matmul.name, 0, 0)
            subgraph.connect(k_weight.name, k_matmul.name, 0, 1)
            if k_bias is not None:
                subgraph.connect(k_bias.name, k_matmul.name, 0, 2)

            k_data = Data(name=f"{self.name}_k", shape=None, dtype=None)
            subgraph.add_node(k_data)
            subgraph.connect(k_matmul.name, k_data.name, 0, 0)

            # V projection
            v_matmul = MatMul(
                name=f"{self.name}_v_proj",
                attrs={
                    "in_channels": input_dim,
                    "out_channels": head_dim,
                    "batch_size": seq_len,
                    "input_dtype": DataType.BF16,
                    "weight_dtype": DataType.BF16,
                },
            )
            subgraph.add_node(v_matmul)
            subgraph.connect(input_buffer.name, v_matmul.name, 0, 0)
            subgraph.connect(v_weight.name, v_matmul.name, 0, 1)
            if v_bias is not None:
                subgraph.connect(v_bias.name, v_matmul.name, 0, 2)

            v_data = Data(name=f"{self.name}_v", shape=None, dtype=None)
            subgraph.add_node(v_data)
            subgraph.connect(v_matmul.name, v_data.name, 0, 0)
        else:
            # Use input directly as Q/K/V (assume they're already projected)
            q_data = input_buffer
            k_data = input_buffer
            v_data = input_buffer

        # Transpose K for attention computation: (seq_len, head_dim) -> (head_dim, seq_len)
        k_transpose_view = create_reshape_view(
            name=f"{self.name}_k_transpose",
            target_shape=(head_dim, seq_len),
        )
        subgraph.add_node(k_transpose_view)
        subgraph.connect(k_data.name, k_transpose_view.name)

        # Compute attention scores: Q @ K^T -> (seq_len, seq_len)
        scores_matmul = MatMul(
            name=f"{self.name}_scores",
            attrs={
                "in_channels": head_dim,
                "out_channels": seq_len,
                "batch_size": seq_len,
                "input_dtype": DataType.BF16,
                "weight_dtype": DataType.BF16,
            },
        )
        subgraph.add_node(scores_matmul)
        subgraph.connect(q_data.name, scores_matmul.name, 0, 0)
        subgraph.connect(k_transpose_view.name, scores_matmul.name, 0, 1)

        scores_data = Data(name=f"{self.name}_scores_out", shape=None, dtype=None)
        subgraph.add_node(scores_data)
        subgraph.connect(scores_matmul.name, scores_data.name, 0, 0)

        # Scale by scaler (default: 1/sqrt(head_dim))
        scale_op = Multiply(
            name=f"{self.name}_scale",
            attrs={
                "output_dtype": DataType.BF16,
                "bc_mode": 2,
                "mult_or_div": 0,
                "scalar": scaler,
            },
        )
        subgraph.add_node(scale_op)
        subgraph.connect(scores_data.name, scale_op.name, 0, 0)

        scaled_scores = Data(name=f"{self.name}_scaled_scores", shape=None, dtype=None)
        subgraph.add_node(scaled_scores)
        subgraph.connect(scale_op.name, scaled_scores.name, 0, 0)

        # Apply softmax to get attention weights
        softmax = Softmax(
            name=f"{self.name}_softmax",
            attrs={
                "axis": -1,
                "feature_dim": seq_len,
            },
        )
        subgraph.add_node(softmax)
        subgraph.connect(scaled_scores.name, softmax.name, 0, 0)

        attn_weights = Data(name=f"{self.name}_attn_weights", shape=None, dtype=None)
        subgraph.add_node(attn_weights)
        subgraph.connect(softmax.name, attn_weights.name, 0, 0)

        # Compute output: attention_weights @ V -> (seq_len, head_dim)
        output_matmul = MatMul(
            name=f"{self.name}_output",
            attrs={
                "in_channels": seq_len,
                "out_channels": head_dim,
                "batch_size": seq_len,
                "input_dtype": DataType.BF16,
                "weight_dtype": DataType.BF16,
            },
        )
        subgraph.add_node(output_matmul)
        subgraph.connect(attn_weights.name, output_matmul.name, 0, 0)
        subgraph.connect(v_data.name, output_matmul.name, 0, 1)

        output_data = Data(name=f"{self.name}_output_data", shape=None, dtype=None)
        subgraph.add_node(output_data)
        subgraph.connect(output_matmul.name, output_data.name, 0, 0)

        return {
            "subgraph": subgraph,
            "input_mapping": {0: (input_buffer.name, None)},
            "output_mapping": {0: (output_data.name, 0)},
        }
