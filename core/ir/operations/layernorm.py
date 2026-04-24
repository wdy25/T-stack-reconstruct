from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from core.ir.data import Data, DataType
from core.ir.operation import Operation
from core.ir.graph import Graph
from core.ir.operations.mat_mul import MatMul
from core.ir.operations.add import Add
from core.ir.operations.multiply import Multiply
from core.ir.operations.nonlinear import Nonlinear


class LayerNorm(Operation):
    """Layer Normalization composed from existing primitive operations.

    Ports:
        inputs:
            - 0: input data(2D tensor: (batch_size, normalized_shape))
        outputs:
            - 0: output data(2D tensor: (batch_size, normalized_shape) is the same as inputs[0])

    Required attrs:
        - normalized_shape (Tuple[int,]): shape of the last dimensions being normalized, currently only assumed to be 1D.
        - epsilon (float): small constant for numerical stability.

    Optional attrs (if provided, they override created constant nodes):
        - elementwise_affine (bool): whether to apply elementwise affine transform. Defaults to False.
            - gamma (float): scale parameter value. Required when elementwise_affine is True.
            - beta (float): bias parameter value. Required when elementwise_affine is True.
                Note: elementwise_affine=True is currently not implemented.

    This composite operation expands to the following sequence when converted to primitives:
        1. Reduce-mean over the last dimension using MatMul with a constant vector.
        2. Subtract the mean from inputs.
        3. Compute variance via elementwise square and another reduction.
        4. Add epsilon, take square root (Nonlinear sqrt) to obtain std.
        5. Divide normalized tensor by std, then apply affine (gamma/beta).
    """

    def __init__(self, name: str, attrs: Optional[Dict[str, object]] = None) -> None:
        super().__init__(name, attrs)
        required_attrs = ["normalized_shape", "epsilon"]
        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for LayerNorm operation.")

        if not isinstance(self.attrs["normalized_shape"], tuple):
            raise ValueError("'normalized_shape' must be a tuple of integers.")
        normalized_shape = self.attrs["normalized_shape"]
        if len(normalized_shape) != 1:
            raise NotImplementedError("Current LayerNorm implementation assumes 1D normalized shape.")
        
        if not isinstance(self.attrs["epsilon"], (float, int)):
            raise ValueError("'epsilon' must be a float.")

        self.attrs.setdefault("elementwise_affine", False)

        elementwise_affine = self.attrs["elementwise_affine"]
        if not isinstance(elementwise_affine, bool):
            raise ValueError("'elementwise_affine' must be a boolean.")

        if elementwise_affine:
            if "gamma" not in self.attrs or "beta" not in self.attrs:
                raise ValueError("'gamma' and 'beta' must be provided when 'elementwise_affine' is True.")
            if not isinstance(self.attrs["gamma"], float) or not isinstance(self.attrs["beta"], float):
                raise ValueError("'gamma' and 'beta' must be floats when 'elementwise_affine' is True.")
            raise NotImplementedError("LayerNorm with elementwise_affine=True is not implemented yet.")
        else:
            self.attrs.setdefault("gamma", None)
            self.attrs.setdefault("beta", None)

        self.primitive = False # Marking as a non-primitive operation

    def infer(self, inputs: Dict[int, Data]) -> list[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        if 0 not in inputs:
            raise ValueError("LayerNorm expects input tensor on port 0.")

        input_data = inputs[0]

        normalized_shape = self.attrs["normalized_shape"]
        if input_data.shape[-len(normalized_shape):] != normalized_shape:
            raise ValueError("Input's trailing dimensions must match 'normalized_shape'.")

        return [(input_data.shape, input_data.dtype)]

    def to_prim(self):
        subgraph = Graph()
        # currently only 1D normalized_shape is supported
        normalized_shape = self.attrs["normalized_shape"]
        feature_dim = normalized_shape[0]

        epsilon = float(self.attrs["epsilon"])

        # Create weight and bias used for sum computation
        sum_weight_shape = (feature_dim, feature_dim)
        sum_weight_tensor = torch.ones(sum_weight_shape, dtype=torch.bfloat16)
        sum_weight = Data(
            name=f"{self.name}_sum_weight",
            shape=sum_weight_shape,
            dtype=DataType.BF16,
            payload=sum_weight_tensor,
        )
        sum_bias_shape = (feature_dim, )
        sum_bias_tensor = torch.zeros(sum_bias_shape, dtype=torch.bfloat16)
        sum_bias = Data(
            name=f"{self.name}_sum_bias",
            shape=sum_bias_shape,
            dtype=DataType.BF16,
            payload=sum_bias_tensor,
        )
        subgraph.add_node(sum_weight)
        subgraph.add_node(sum_bias)

        reduce_sum = MatMul(
            name=f"{self.name}_reduce_sum",
            attrs={
                "in_channels": feature_dim,
                "out_channels": feature_dim,
                "input_dtype": DataType.BF16,
                "weight_dtype": DataType.BF16,
            },
        )
        subgraph.add_node(reduce_sum)
        subgraph.connect(sum_weight.name, reduce_sum.name, 0, 1)
        subgraph.connect(sum_bias.name, reduce_sum.name, 0, 2)

        sum_data = Data(name=f"{self.name}_sum")
        subgraph.add_node(sum_data)
        subgraph.connect(reduce_sum.name, sum_data.name, 0, 0)

        mean_div = Multiply(
            name=f"{self.name}_mean_div",
            attrs={
                "output_dtype": DataType.BF16,
                "bc_mode": 2,
                "mult_or_div": 1,
                "scalar": float(feature_dim),
            },
        )
        subgraph.add_node(mean_div)
        subgraph.connect(sum_data.name, mean_div.name, 0, 0)

        mean_data = Data(name=f"{self.name}_mean")
        subgraph.add_node(mean_data)
        subgraph.connect(mean_div.name, mean_data.name, 0, 0)

        centered_add = Add(
            name=f"{self.name}_centered",
            attrs={
                "output_dtype": DataType.BF16,
                "bc_mode": 0,
                "add_or_sub": 1,
                "scalar": 0.0,
            },
        )
        subgraph.add_node(centered_add)
        # The first input of `centered_add` will be connected externally via `input_mapping`
        subgraph.connect(mean_data.name, centered_add.name, 0, 1)

        centered = Data(name=f"{self.name}_centered_out")
        subgraph.add_node(centered)
        subgraph.connect(centered_add.name, centered.name, 0, 0)

        square_mul = Multiply(
            name=f"{self.name}_square",
            attrs={
                "output_dtype": DataType.BF16,
                "bc_mode": 0,
                "mult_or_div": 0,
                "scalar": 1.0,
            },
        )
        subgraph.add_node(square_mul)
        subgraph.connect(centered.name, square_mul.name, 0, 0)
        subgraph.connect(centered.name, square_mul.name, 0, 1)

        squared = Data(name=f"{self.name}_squared")
        subgraph.add_node(squared)
        subgraph.connect(square_mul.name, squared.name, 0, 0)

        var_reduce = MatMul(
            name=f"{self.name}_var_reduce",
            attrs={
                "in_channels": feature_dim,
                "out_channels": feature_dim,
                "input_dtype": DataType.BF16,
                "weight_dtype": DataType.BF16,
            },
        )
        subgraph.add_node(var_reduce)
        subgraph.connect(squared.name, var_reduce.name, 0, 0)
        subgraph.connect(sum_weight.name, var_reduce.name, 0, 1)
        subgraph.connect(sum_bias.name, var_reduce.name, 0, 2)

        variance_sum = Data(name=f"{self.name}_variance_sum")
        subgraph.add_node(variance_sum)
        subgraph.connect(var_reduce.name, variance_sum.name, 0, 0)

        variance_div = Multiply(
            name=f"{self.name}_variance_div", 
            attrs={
                "output_dtype": DataType.BF16,
                "bc_mode": 2,
                "mult_or_div": 1,
                "scalar": float(feature_dim),
            },
        )
        subgraph.add_node(variance_div)
        subgraph.connect(variance_sum.name, variance_div.name, 0, 0)

        variance = Data(name=f"{self.name}_variance")
        subgraph.add_node(variance)
        subgraph.connect(variance_div.name, variance.name, 0, 0)

        eps_add = Add(
            name=f"{self.name}_eps_add",
            attrs={
                "output_dtype": DataType.BF16,
                "bc_mode": 2,
                "add_or_sub": 0,
                "scalar": epsilon,
            },
        )
        subgraph.add_node(eps_add)
        subgraph.connect(variance.name, eps_add.name, 0, 0)

        variance_eps = Data(name=f"{self.name}_variance_eps")
        subgraph.add_node(variance_eps)
        subgraph.connect(eps_add.name, variance_eps.name, 0, 0)

        std_sqrt = Nonlinear(
            name=f"{self.name}_std_sqrt",
            attrs={
                "output_dtype": DataType.BF16,
                "function": "sqrt",
            },
        )
        subgraph.add_node(std_sqrt)
        subgraph.connect(variance_eps.name, std_sqrt.name, 0, 0)

        std = Data(name=f"{self.name}_std")
        subgraph.add_node(std)
        subgraph.connect(std_sqrt.name, std.name, 0, 0)

        normalize_div = Multiply(
            name=f"{self.name}_normalize",
            attrs={
                "output_dtype": DataType.BF16,
                "bc_mode": 0,
                "mult_or_div": 1,
                "scalar": 1.0,
            },
        )
        subgraph.add_node(normalize_div)
        subgraph.connect(centered.name, normalize_div.name, 0, 0)
        subgraph.connect(std.name, normalize_div.name, 0, 1)

        # Apply affine transformation
        '''
        gamma_node = self.attrs["gamma"] or f"{self.name}_gamma"
        beta_node = self.attrs["beta"] or f"{self.name}_beta"

        gamma_tensor = torch.ones((1, feature_dim), dtype=torch.bfloat16)
        beta_tensor = torch.zeros((1, feature_dim), dtype=torch.bfloat16)
        gamma_data = Data(name=gamma_node, shape=(1, feature_dim), dtype=DataType.BF16, payload=gamma_tensor)
        beta_data = Data(name=beta_node, shape=(1, feature_dim), dtype=DataType.BF16, payload=beta_tensor)
        subgraph.add_node(gamma_data)
        subgraph.add_node(beta_data)

        scale_mul = Multiply(
            name=f"{self.name}_scale",
            attrs={
                "output_dtype": DataType.BF16,
                "bc_mode": 1,
                "mult_or_div": 0,
                "scalar": 1.0,
            },
        )
        subgraph.add_node(scale_mul)
        subgraph.connect(normalized.name, scale_mul.name, 0, 0)
        subgraph.connect(gamma_data.name, scale_mul.name, 0, 1)

        scaled = Data(name=f"{self.name}_scaled")
        subgraph.add_node(scaled)
        subgraph.connect(scale_mul.name, scaled.name, 0, 0)

        shift_add = Add(
            name=f"{self.name}_shift",
            attrs={
                "output_dtype": DataType.BF16,
                "bc_mode": 1,
                "add_or_sub": 0,
                "scalar": 0.0,
            },
        )
        subgraph.add_node(shift_add)
        subgraph.connect(scaled.name, shift_add.name, 0, 0)
        subgraph.connect(beta_data.name, shift_add.name, 0, 1)
        

        output = Data(name=f"{self.name}_output")
        subgraph.add_node(output)
        subgraph.connect(shift_add.name, output.name, 0, 0)
        '''

        return {
            "subgraph": subgraph,
            "input_mapping": {
                0: [(reduce_sum.name, 0), (centered_add.name, 0)],
            },
            "output_mapping": {0: (normalize_div.name, 0)},
        }
