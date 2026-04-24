import math
import torch

from typing import Dict, List, Optional, Tuple

from core.ir.graph import Graph
from core.ir.data import Data, DataType, ViewData
from core.ir.operation import Operation
from core.ir.operations.add import Add
from core.ir.operations.max import Max
from core.ir.operations.transpose import Transpose
from core.ir.operations.mat_mul import MatMul
from core.ir.operations.multiply import Multiply
from core.ir.operations.nonlinear import Nonlinear


class Softmax(Operation):
    """Layer-wise softmax constructed from existing primitive operations.

    Ports:
        inputs:
            - 0: tensor to normalize (shape>=2D, BF16 only)
        outputs:
            - 0: softmax probabilities matching the input shape (the same shape as inputs[0], BF16 only)

    Required attributes:
        - input_shape (Tuple[int, ...]): shape of the input tensor, assert during shape inference.
            - feature_dim (int): the last dimension of input_shape.
    
    Optional attributes(currently can't be changed):
        - axis (int): logical axis along which softmax is computed. currently only supports -1.
        
     When lowered to primitives the operation expands to the following steps:
          1. Reshape the input to a 4D view and transpose it so the feature dimension
              becomes contiguous for per-vector processing.
          2. Collapse the tensor to `(feature_dim, vector_count)` so each column
              represents one softmax vector.
          3. Use `Max` to find the column-wise maximum and improve numerical stability.
          4. Subtract the maximum from every element with a broadcast `Add` to center
              the logits before exponentiation.
          5. Restore the original layout via transpose + reshape so the following
              operations can work on a tensor shaped like the input.
          6. Apply `Nonlinear(exp)` to obtain exponentials of the shifted logits.
          7. Run a `MatMul` with an all-ones weight matrix to both sum along the
              feature dimension and broadcast the denominator back to every position.
          8. Perform element-wise division with `Multiply` configured for reciprocal
              mode to produce the normalized probabilities.
    """

    def __init__(self, name: str, attrs: Optional[Dict[str, object]] = None) -> None:
        super().__init__(name, attrs)
        required_attrs = ["input_shape"]
        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for Softmax operation.")
            
        input_shape = self.attrs["input_shape"]
        self.attrs["feature_dim"] = input_shape[-1]

        self.attrs.setdefault("axis", -1)

        axis = self.attrs["axis"]
        if axis != - 1:
            raise NotImplementedError("Softmax currently supports the last dimension only.")
        
        vector_count = math.prod(input_shape[:-1])
        self.attrs["vector_count"] = int(vector_count)
        self.attrs["input_rank"] = len(input_shape)

        self.primitive = False # Marking as a non-primitive operation

    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        if 0 not in inputs:
            raise ValueError("Softmax expects input tensor on port 0.")

        input_data = inputs[0]
        if input_data.shape is None or len(input_data.shape) < 1:
            raise ValueError("Input shape must be known for Softmax.")
        if len(input_data.shape) < 2:
            raise ValueError("Softmax currently requires input tensors with rank >= 2.")

        input_shape = input_data.shape
        if input_shape is None:
            raise ValueError("Softmax requires a concrete input shape.")

        cached_input_shape = self.attrs.get("input_shape")
        if cached_input_shape is None:
            raise ValueError("Softmax requires input_shape attribute to be set before infer.")
        elif cached_input_shape != input_shape:
            raise ValueError("Input shape does not match configured input_shape.")

        if input_data.dtype != DataType.BF16:
            raise ValueError("Softmax expects BF16 input. Insert a Convert op beforehand if needed.")

        return [(input_data.shape, input_data.dtype)]

    def to_prim(self):
        feature_dim = self.attrs.get("feature_dim")
        if feature_dim is None:
            raise ValueError("Softmax requires feature_dim to be resolved before lowering.")

        input_shape: Optional[Tuple[int, ...]] = self.attrs.get("input_shape")
        if input_shape is None:
            raise ValueError("Softmax requires input_shape to be cached during infer.")

        vector_count = int(self.attrs.get("vector_count", 0))
        if vector_count <= 0:
            raise ValueError("Softmax requires a positive vector_count inferred during infer.")

        subgraph = Graph()

        # Step 1: Reshape to (1, 1, vector_count, feature_dim) so we can transpose the feature dim to the front
        input_reshape_4d = ViewData(
            name=f"{self.name}_input_reshape_4d",
            view_type="reshape",
            target_shape=(1, 1, vector_count, feature_dim),
        )
        subgraph.add_node(input_reshape_4d)

        # Still Step 1: transpose the feature dimension to be contiguous for per-vector ops
        transpose_to_feature_first = Transpose(
            name=f"{self.name}_transpose_to_feature_first",
            attrs={
                "dim_A": 1,
                "dim_B": 1,
                "dim_C": vector_count,
                "dim_D": feature_dim,
                "transpose_order": "CD",
            },
        )
        subgraph.add_node(transpose_to_feature_first)
        subgraph.connect(input_reshape_4d.name, transpose_to_feature_first.name, 0, 0)

        transposed_feature_first = Data(name=f"{self.name}_feature_first_4d")
        subgraph.add_node(transposed_feature_first)
        subgraph.connect(transpose_to_feature_first.name, transposed_feature_first.name, 0, 0)

        # Step 2: Collapse to (feature_dim, vector_count) so each column is an independent softmax vector
        feature_first_view = ViewData(
            name=f"{self.name}_feature_first_view",
            view_type="reshape",
            target_shape=(feature_dim, vector_count),
        )
        subgraph.add_node(feature_first_view)
        subgraph.connect(transposed_feature_first.name, feature_first_view.name, 0, 0)

        # Step 3: Find the per-vector max for numerical stability
        max_op = Max(
            name=f"{self.name}_max",
            attrs={"output_dtype": DataType.BF16},
        )
        subgraph.add_node(max_op)
        subgraph.connect(feature_first_view.name, max_op.name, 0, 0)

        max_out = Data(name=f"{self.name}_max_out")
        subgraph.add_node(max_out)
        subgraph.connect(max_op.name, max_out.name, 0, 0)

        # Step 4: Subtract the max via broadcast add (configured as subtraction)
        shift_op = Add(
            name=f"{self.name}_shift",
            attrs={
                "output_dtype": DataType.BF16,
                "bc_mode": 1,
                "add_or_sub": 1,
            },
        )
        subgraph.add_node(shift_op)
        subgraph.connect(feature_first_view.name, shift_op.name, 0, 0)
        subgraph.connect(max_out.name, shift_op.name, 0, 1)

        # shifted_feature_after_sub = Data(name=f"{self.name}_shifted_feature_after_sub")
        # subgraph.add_node(shifted_feature_after_sub)
        # subgraph.connect(shift_op.name, shifted_feature_after_sub.name, 0, 0)

        # # apply a uniform +5 offset after the max subtraction
        # add_offset = Add(
        #     name=f"{self.name}_shift_bias",
        #     attrs={
        #         "output_dtype": DataType.BF16,
        #         "bc_mode": 2,
        #         "add_or_sub": 0,
        #         "scalar": 5.0,
        #     },
        # )
        # subgraph.add_node(add_offset)
        # subgraph.connect(shifted_feature_after_sub.name, add_offset.name, 0, 0)

        shifted_feature_first = Data(name=f"{self.name}_shifted_feature_first")
        subgraph.add_node(shifted_feature_first)
        subgraph.connect(shift_op.name, shifted_feature_first.name, 0, 0)

        # Step 5: Restore the original layout (transpose back + reshape) before exponentiation
        shifted_feature_first_view = ViewData(
            name=f"{self.name}_shifted_feature_first_view",
            view_type="reshape",
            target_shape=(1, 1, feature_dim, vector_count),
        )
        subgraph.add_node(shifted_feature_first_view)
        subgraph.connect(shifted_feature_first.name, shifted_feature_first_view.name, 0, 0)

        transpose_back = Transpose(
            name=f"{self.name}_transpose_back",
            attrs={
                "dim_A": 1,
                "dim_B": 1,
                "dim_C": feature_dim,
                "dim_D": vector_count,
                "transpose_order": "CD",
            },
        )
        subgraph.add_node(transpose_back)
        subgraph.connect(shifted_feature_first_view.name, transpose_back.name, 0, 0)

        shifted_back = Data(name=f"{self.name}_shifted_back_4d")
        subgraph.add_node(shifted_back)
        subgraph.connect(transpose_back.name, shifted_back.name, 0, 0)

        shifted_back_view = ViewData(
            name=f"{self.name}_shifted_back_view",
            view_type="reshape",
            target_shape=input_shape,
        )
        subgraph.add_node(shifted_back_view)
        subgraph.connect(shifted_back.name, shifted_back_view.name, 0, 0)

        # Step 6: Apply element-wise exponentiation to the shifted logits
        exp_op = Nonlinear(
            name=f"{self.name}_exp",
            attrs={
                "output_dtype": DataType.BF16,
                "function": "exp",
            },
        )
        subgraph.add_node(exp_op)
        subgraph.connect(shifted_back_view.name, exp_op.name, 0, 0)

        exp_out = Data(name=f"{self.name}_exp_out")
        subgraph.add_node(exp_out)
        subgraph.connect(exp_op.name, exp_out.name, 0, 0)

        # Step 7: Use a (feature_dim, feature_dim) all-ones MatMul to sum and broadcast denominators
        sum_expand_weight_shape = (feature_dim, feature_dim)
        sum_expand_weight_tensor = torch.ones(sum_expand_weight_shape, dtype=torch.bfloat16)
        sum_expand_weight = Data(
            name=f"{self.name}_sum_expand_weight",
            shape=sum_expand_weight_shape,
            dtype=DataType.BF16,
            payload=sum_expand_weight_tensor,
        )
        subgraph.add_node(sum_expand_weight)
        sum_expand_bias_shape = (feature_dim,)
        sum_expand_bias_tensor = torch.zeros(sum_expand_bias_shape, dtype=torch.bfloat16)
        sum_expand_bias = Data(
            name=f"{self.name}_sum_expand_bias",
            shape=sum_expand_bias_shape,
            dtype=DataType.BF16,
            payload=sum_expand_bias_tensor,
        )
        subgraph.add_node(sum_expand_bias)

        sum_expand = MatMul(
            name=f"{self.name}_sum_expand",
            attrs={
                "in_channels": feature_dim,
                "out_channels": feature_dim,
                "input_dtype": DataType.BF16,
                "weight_dtype": DataType.BF16,
            },
        )
        subgraph.add_node(sum_expand)
        subgraph.connect(exp_out.name, sum_expand.name, 0, 0)
        subgraph.connect(sum_expand_weight.name, sum_expand.name, 0, 1)
        subgraph.connect(sum_expand_bias.name, sum_expand.name, 0, 2)

        expanded = Data(name=f"{self.name}_expanded")
        subgraph.add_node(expanded)
        subgraph.connect(sum_expand.name, expanded.name, 0, 0)

        # Step 8: Divide exponentials by the normalization denominator using reciprocal Multiply
        divide = Multiply(
            name=f"{self.name}_divide",
            attrs={
                "output_dtype": DataType.BF16,
                "bc_mode": 0,
                "mult_or_div": 1,
                "scalar": 1.0,
            },
        )
        subgraph.add_node(divide)
        subgraph.connect(exp_out.name, divide.name, 0, 0)
        subgraph.connect(expanded.name, divide.name, 0, 1)

        return {
            "subgraph": subgraph,
            "input_mapping": {0: (input_reshape_4d.name, 0)},
            "output_mapping": {0: (divide.name, 0)},
        }
