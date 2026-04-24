import math
from typing import Dict, List, Optional, Tuple

from core.ir.graph import Graph
from core.ir.operation import Operation
from core.ir.data import (
    ConcatData,
    Data,
    DataType,
    create_reshape_view,
    create_slice_view,
    create_reshape_and_slice_view,
)
from core.ir.operations.mat_mul import MatMul
from core.ir.operations.add import Add


class MatMulOne2Many(Operation):
    """Matrix multiplication that expands one input tensor across multiple weight slices.

    Ports:
        inputs:
            - 0: input feature tensor, 2D (batch_size, C_in).
            - 1: weight tensor, 3D (dim, C_in, C_out).
            - 2: bias tensor, 2D (dim, C_out).
        outputs:
            - 0: output feature tensor, 3D (dim, batch_size, C_out).
                note: the output is already Data node, needs to connect operation node to it.

    Attributes:
        required attrs (Dict[str, Any]):
            - 'dim' (int): number of weight slices.
            - 'in_channels' (int): input channel count C_in.
            - 'out_channels' (int): output channel count C_out.
            - 'batch_size' (int): input batch size.
        optional attrs:
            - 'shape' (Tuple[int, ...]): define output shape explicitly (last dimension must equal C_out).
                - note: math.prod(shape) must equal dim * batch_size * C_out if provided.

    Architecture:
        This operator is non-primitive. The ``to_prim`` method constructs a subgraph that:
            1. uses ``create_reshape_view`` to expand the input to (1, batch_size, C_in);
            2. creates per-slice weight/bias views using ``create_slice_view`` or
               ``create_reshape_and_slice_view``;
            3. instantiates a ``MatMul`` primitive for each slice and connects the views;
            4. concatenates the ``dim`` outputs with ``ConcatData`` into the final result.
    """

    def __init__(self, name: str, attrs: Optional[Dict[str, object]] = None) -> None:
        super().__init__(name, attrs)
        required_attrs = ["dim", "in_channels", "out_channels", "batch_size"]
        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for MatMulOne2Many operation.")

        if isinstance(self.attrs["dim"], int) and self.attrs["dim"] <= 1:    
            raise ValueError("Attribute 'dim' must be greater than 1 for MatMulOne2Many operation.")
        if not isinstance(self.attrs["in_channels"], int) or self.attrs["in_channels"] <= 0:
            raise ValueError("Attribute 'in_channels' must be a positive integer for MatMulOne2Many operation.")
        if not isinstance(self.attrs["out_channels"], int) or self.attrs["out_channels"] <= 0:
            raise ValueError("Attribute 'out_channels' must be a positive integer for MatMulOne2Many operation.")
        if not isinstance(self.attrs["batch_size"], int) or self.attrs["batch_size"] <= 0:
            raise ValueError("Attribute 'batch_size' must be a positive integer for MatMulOne2Many operation.") 

        shape_attr = self.attrs.get("shape")
        if shape_attr is not None:
            if not isinstance(shape_attr, tuple) or len(shape_attr) < 2:
                raise ValueError("Attribute 'shape' must be a tuple with at least two positive integers.")
            if not all(isinstance(dim, int) and dim > 0 for dim in shape_attr):
                raise ValueError("All dimensions in 'shape' must be positive integers.")
            if shape_attr[-1] != self.attrs["out_channels"]:
                raise ValueError("The last dimension of 'shape' must equal 'out_channels'.")
            expected_elements = self.attrs["dim"] * self.attrs["batch_size"] * self.attrs["out_channels"]
            if math.prod(shape_attr) != expected_elements:
                raise ValueError(
                    "Attribute 'shape' total elements do not match dim * batch_size * out_channels."
                )

        self.primitive = False # Marking as a non-primitive operation

    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        if 0 not in inputs:
            raise ValueError("MatMulOne2Many expects input tensor on port 0.")
        if 1 not in inputs:
            raise ValueError("MatMulOne2Many expects weight tensor on port 1.")
        if 2 not in inputs:
            raise ValueError("MatMulOne2Many expects bias tensor on port 2.")

        input_data = inputs[0]
        weight_data = inputs[1]
        bias_data = inputs[2]

        if weight_data.dtype != input_data.dtype:
            raise ValueError("Weight dtype must match input dtype.")
        if bias_data.dtype != input_data.dtype:
            raise ValueError("Bias dtype must match input dtype.")

        input_shape = input_data.shape
        weight_shape = weight_data.shape
        bias_shape = bias_data.shape

        if input_shape is None or len(input_shape) != 2:
            raise ValueError("Input data must have shape (batch_size, C_in).")
        if weight_shape is None or len(weight_shape) != 3:
            raise ValueError("Weight data must have shape (dim, C_in, C_out).")
        if bias_shape is None or len(bias_shape) != 2:
            raise ValueError("Bias data must have shape (dim, C_out).")

        if input_shape[0] != self.attrs["batch_size"]:
            raise ValueError("Input batch size does not match 'batch_size' attribute.")
        if input_shape[1] != self.attrs["in_channels"]:
            raise ValueError("Input C_in does not match 'in_channels' attribute.")

        if weight_shape[0] != self.attrs["dim"]:
            raise ValueError("Weight dim does not match 'dim' attribute.")
        if weight_shape[1] != input_shape[1]:
            raise ValueError("Input C_in does not match weight C_in.")
        if weight_shape[2] != self.attrs["out_channels"]:
            raise ValueError("Weight C_out does not match 'out_channels' attribute.")
        
        if bias_shape[0] != self.attrs["dim"] or bias_shape[1] != self.attrs["out_channels"]:
            raise ValueError("Bias shape must be (dim, out_channels).")

        self.inputs_dtype = input_data.dtype

        output_shape = self.attrs.get("shape")
        if output_shape is None:
            output_shape = (weight_shape[0], input_shape[0], weight_shape[2])
        return [(output_shape, self.inputs_dtype)]

    def to_prim(self):
        dim = self.attrs["dim"]
        batch_size = self.attrs["batch_size"]
        c_in = self.attrs["in_channels"]
        c_out = self.attrs["out_channels"]
        
        subgraph = Graph()

        concat_shape = self.attrs.get("shape") or (dim, batch_size, c_out)

        concat_node = ConcatData(
            name=f"{self.name}_output",
            num_inputs=dim,
            shape=concat_shape,
        )
        subgraph.add_node(concat_node)

        input_mapping = {
            0: [], # input
            1: [], # weight
            2: [], # bias
        }

        for idx in range(dim):
            weight_view = create_reshape_and_slice_view(
                name=f"{self.name}_weight_slice_{idx}",
                target_shape=(dim*c_in, c_out),
                pre_idx=[],
                slice_idx=0,
                slice_start=idx * c_in,
                slice_end=(idx + 1) * c_in,
            )
            subgraph.add_node(weight_view)
            input_mapping[1].append((weight_view.name, 0))

            bias_view = create_slice_view(
                name=f"{self.name}_bias_slice_{idx}",
                pre_idx=[],
                slice_idx=0,
                slice_start=idx,
                slice_end=(idx + 1),
            )
            subgraph.add_node(bias_view)
            input_mapping[2].append((bias_view.name, 0))

            matmul_op = MatMul(
                name=f"{self.name}_matmul_{idx}",
                attrs={
                    "batch_size": batch_size,
                    "in_channels": c_in,
                    "out_channels": c_out,
                },
            )
            subgraph.add_node(matmul_op)
            input_mapping[0].append((matmul_op.name, 0))
            subgraph.connect(weight_view.name, matmul_op.name, 0, 1)
            subgraph.connect(bias_view.name, matmul_op.name, 0, 2)

            slice_mat_mul_output = Data(name=f"{self.name}_slice_mat_mul_out_{idx}")
            subgraph.add_node(slice_mat_mul_output)
            subgraph.connect(matmul_op.name, slice_mat_mul_output.name, 0, 0)
            subgraph.connect(slice_mat_mul_output.name, concat_node.name, 0, idx)

        return {
            "subgraph": subgraph,
            "input_mapping": input_mapping,
            "output_mapping": {0: (concat_node.name, 0)}, # note: the output(concat_node) is already a data node, note a operation node.
        }
