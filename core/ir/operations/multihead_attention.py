from typing import Dict, List, Optional, Tuple

import torch

from core.ir.graph import Graph
from core.ir.data import (
    Data,
    DataType,
    create_reshape_view,
)
from core.ir.operation import Operation
from core.ir.operations.mat_mul import MatMul
from core.ir.operations.mat_mul_one2many import MatMulOne2Many
from core.ir.operations.multiply import Multiply
from core.ir.operations.softmax import Softmax
from core.ir.operations.transpose import Transpose


class MultiheadAttention(Operation):
    """Transformer Multi-Head Attention using multiple SingleheadAttention instances.

    Ports:
        inputs:
            - 0: Input tensor for multi-head attention (3D tensor: (batch_size, seq_len, embed_dim))
            - 1: Q projection weight (3D tensor: (heads_num, embed_dim, head_dim))
            - 2: K projection weight (3D tensor: (heads_num, embed_dim, head_dim))
            - 3: V projection weight (3D tensor: (heads_num, embed_dim, head_dim))
            - 4: Output projection weight to transform concatenated heads (2D tensor: (inner_dim, embed_dim), inner_dim = heads_num * head_dim)
            - 5: Q projection bias (2D tensor: (heads_num, head_dim))
            - 6: K projection bias (2D tensor: (heads_num, head_dim))
            - 7: V projection bias (2D tensor: (heads_num, head_dim))
            - 8: Output projection bias for final output projection (1D tensor: (embed_dim,))
        outputs:
            - 0: Output tensor (3D tensor: (batch_size, seq_len, embed_dim))
                - if you want to get 2D tensor (batch_size*seq_len, embed_dim), you need to make once reshape to make the 3D tensor become 2D.

    Required attrs:
        - embed_dim (int): Total dimension of the model embedding.
        - heads_num (int): Number of attention heads. embed_dim doesn't have to be divisible by heads_num.
        - head_dim (int): Dimension of each attention head. embed_dim/heads_num doesn't have to equal embed_dim.

    Optional attrs:
        - o_weight_name (str): Name for the output projection weight. Defaults to "{name}_o_weight".
        - o_bias_name (str): Name for the output projection bias. Defaults to "{name}_o_bias".

    This implementation creates heads_num separate SingleheadAttention operations,
    each with its own projection weights. The outputs are then concatenated and
    projected through a final output layer.

    Architecture:
        1. For each head i in [0, heads_num):
           - Create SingleheadAttention with dedicated Q/K/V weights
           - Each head projects: input (seq_len, embed_dim) -> (seq_len, head_dim)
        2. Concatenate all head outputs along feature dimension
        3. Apply final output projection: (seq_len, embed_dim) -> (seq_len, embed_dim)
    """

    def __init__(self, name: str, attrs: Optional[Dict[str, object]] = None) -> None:
        super().__init__(name, attrs)
        required_attrs = ["embed_dim", "heads_num", "head_dim"]
        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for MultiheadAttention operation.")

        if not isinstance(self.attrs["embed_dim"], int) or self.attrs["embed_dim"] <= 0:
            raise ValueError("'embed_dim' must be a positive integer.")
        if not isinstance(self.attrs["heads_num"], int) or self.attrs["heads_num"] <= 0:
            raise ValueError("'heads_num' must be a positive integer.")
        if not isinstance(self.attrs["head_dim"], int) or self.attrs["head_dim"] <= 0:
            raise ValueError("'head_dim' must be a positive integer.")
        
        # Weight names - each head will have its own set
        # Head i will use: {name}_head{i}_q_weight, etc.
        # self.attrs.setdefault("o_weight_name", f"{name}_o_weight")
        # self.attrs.setdefault("o_bias_name", f"{name}_o_bias")

        self._cached_batch_size: Optional[int] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_head_dim: Optional[int] = None

        self.primitive = False # Marking as a non-primitive operation

    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        # The operator currently accepts a 3D BF16 tensor on port 0.
        if 0 not in inputs:
            raise ValueError("MultiheadAttention expects input tensor on port 0.")
        required_ports = {
            1: "Q projection weight",
            2: "K projection weight",
            3: "V projection weight",
            4: "output projection weight",
            5: "Q projection bias",
            6: "K projection bias",
            7: "V projection bias",
            8: "output projection bias",
        }
        for port_idx, desc in required_ports.items():
            if port_idx not in inputs:
                raise ValueError(f"MultiheadAttention expects {desc} tensor on port {port_idx}.")
        
        embed_dim = self.attrs["embed_dim"]
        heads_num = self.attrs["heads_num"]
        head_dim = self.attrs["head_dim"]

        input_data = inputs[0]
        input_shape = input_data.shape
        if input_shape is None:
            raise ValueError("Input data shape must be defined for MultiheadAttention.")
        if len(input_shape) != 3:
            raise ValueError(
                "Input data must be 3D (batch_size, sequence_len, embed_dim)."
            )
        if input_shape[-1] != embed_dim:
            raise ValueError(
                f"Input last dimension {input_shape[-1]} does not match configured embed_dim {embed_dim}."
            )
        if input_data.dtype != DataType.BF16:
            raise ValueError("MultiheadAttention currently supports only BF16 inputs.")

        for projection_name, port_idx in {"Q": 1, "K": 2, "V": 3}.items():
            weight_data = inputs[port_idx]
            weight_shape = weight_data.shape
            if weight_shape is None:
                raise ValueError(f"{projection_name} projection weight shape must be defined for MultiheadAttention.")
            if len(weight_shape) != 3:
                raise ValueError(
                    f"{projection_name} projection weight must be 3D (heads_num, embed_dim, head_dim)."
                )
            if weight_shape[0] != heads_num:
                raise ValueError(
                    f"{projection_name} projection weight heads_num {weight_shape[0]} does not match configured heads_num {heads_num}."
                )
            if weight_shape[1] != embed_dim:
                raise ValueError(
                    f"{projection_name} projection weight embed_dim {weight_shape[1]} does not match configured embed_dim {embed_dim}."
                )
            if weight_shape[2] != head_dim:
                raise ValueError(
                    f"{projection_name} projection weight head_dim {weight_shape[2]} does not match configured head_dim {head_dim}."
                )
            if weight_data.dtype != DataType.BF16:
                raise ValueError(f"{projection_name} projection weight must use BF16 dtype.")

        weight_o_data = inputs[4]
        weight_o_shape = weight_o_data.shape
        if weight_o_shape is None:
            raise ValueError("Output Projection Weight data shape must be defined for MultiheadAttention.")
        if len(weight_o_shape) != 2:
            raise ValueError(
                "Output Projection Weight data must be 2D (inner_dim, embed_dim)."
            )
        if weight_o_shape[-2] != heads_num*head_dim:
            raise ValueError(
                f"Output Projection Weight inner_dim {weight_o_shape[-2]} does not match configured head_dim*{head_dim}."
            )
        if weight_o_shape[-1] != embed_dim:
            raise ValueError(
                f"Output Projection Weight embed_dim {weight_o_shape[-1]} does not match configured embed_dim {embed_dim}."
            )
        if weight_o_data.dtype != DataType.BF16:
            raise ValueError("Output projection weight must use BF16 dtype.")
        
        for projection_name, port_idx in {"Q": 5, "K": 6, "V": 7}.items():
            bias_data = inputs[port_idx]
            bias_shape = bias_data.shape
            if bias_shape is None:
                raise ValueError(f"{projection_name} projection bias shape must be defined for MultiheadAttention.")
            if len(bias_shape) != 2:
                raise ValueError(
                    f"{projection_name} projection bias must be 2D (heads_num, head_dim)."
                )
            if bias_shape[0] != heads_num:
                raise ValueError(
                    f"{projection_name} projection bias heads_num {bias_shape[0]} does not match configured heads_num {heads_num}."
                )
            if bias_shape[1] != head_dim:
                raise ValueError(
                    f"{projection_name} projection bias head_dim {bias_shape[1]} does not match configured head_dim {head_dim}."
                )
            if bias_data.dtype != DataType.BF16:
                raise ValueError(f"{projection_name} projection bias must use BF16 dtype.")

        bias_o_data = inputs[8]
        bias_o_shape = bias_o_data.shape
        if bias_o_shape is None:
            raise ValueError("Output Projection Bias data shape must be defined for MultiheadAttention.")
        if len(bias_o_shape) != 1:
            raise ValueError(
                "Output Projection Bias data must be 1D (embed_dim,)."
            )
        if bias_o_shape[0] != embed_dim:
            raise ValueError(
                f"Output Projection Bias length {bias_o_shape[0]} does not match configured embed_dim {embed_dim}."
            )
        if bias_o_data.dtype != DataType.BF16:
            raise ValueError("Output projection bias must use BF16 dtype.")

        # Cache values required for to_prim
        self._cached_batch_size = input_shape[0]
        self._cached_seq_len = input_shape[1]
        self._cached_head_dim = head_dim

        return [(input_data.shape, input_data.dtype)]

    def to_prim(self):
        """Convert to primitive operations subgraph.

        Decomposes multi-head attention into basic operations following this pipeline:
        1. Reshape input from 3D (B, n, embed_dim) to 2D (B*n, embed_dim)
        2. Apply MatMulOne2Many with provided Q/K/V weights and biases -> 3D (heads_num, B*n, head_dim) -> reshape internally 3D (heads_num*B, n, head_dim)
        3. Transpose K to (heads_num*B, head_dim, n)
        4. MatMul Q @ K^T -> (heads_num*B, n, n)
        5. Multiply by scale factor (head_dim ** -0.5)
        6. Apply Softmax
        7. MatMul attention @ V -> (heads_num*B, n, head_dim)
        8. Reshape to 4D (heads_num, B, n, head_dim)
        9. Transpose to 4D (B, n, heads_num, head_dim)
        10. Reshape to 3D (B, n, heads_num*head_dim)
        11. MatMul with output projection -> 3D (B, n, embed_dim)
        """
        if self._cached_batch_size is None or self._cached_seq_len is None:
            raise ValueError("infer must be called before to_prim for MultiheadAttention.")

        batch_size = self._cached_batch_size
        seq_len = self._cached_seq_len
        head_dim = int(self.attrs["head_dim"])
        embed_dim = int(self.attrs["embed_dim"])
        heads_num = int(self.attrs["heads_num"])
        inner_dim = heads_num * head_dim
        scale_factor = head_dim ** -0.5

        subgraph = Graph()

        # Step 1: Reshape input from 3D (B, n, embed_dim) to 2D (B*n, embed_dim)
        input_reshape_2d = create_reshape_view(
            name=f"{self.name}_input_reshape",
            target_shape=(batch_size * seq_len, embed_dim),
        )
        subgraph.add_node(input_reshape_2d)

        # Step 2: Apply MatMulOne2Many for Q, K, V projections using per-head parameters and Reshape Q, V from 3D (heads_num, B*n, head_dim) to 3D (heads_num*B, n, head_dim) and K from 3D (heads_num, B*n, head_dim) to 4D (1, heads_num*B, n, head_dim) for transpose
        qkv_names = ["Q", "K", "V"]
        qkv_projects: Dict[str, MatMulOne2Many] = {}
        for qkv_name in qkv_names:
            matmul_one2many = MatMulOne2Many(
                name=f"{self.name}_{qkv_name}_proj",
                attrs={
                    "dim": heads_num,
                    "in_channels": embed_dim,
                    "out_channels": head_dim,
                    "batch_size": batch_size * seq_len,
                    "shape": (heads_num*batch_size, seq_len, head_dim) if qkv_name != "K" else (1, heads_num*batch_size, seq_len, head_dim),
                },
            )
            subgraph.add_node(matmul_one2many)
            subgraph.connect(input_reshape_2d.name, matmul_one2many.name, 0, 0)

            qkv_projects[qkv_name] = matmul_one2many

        Q_projected = qkv_projects["Q"]
        K_projected = qkv_projects["K"]
        V_projected = qkv_projects["V"]

        # Step 3a: Transpose K from 4D (1, heads_num*B, n, head_dim) to 4D (1, heads_num*B, head_dim, n)
        K_transpose = Transpose(
            name=f"{self.name}_K_transpose",
            attrs={
                "dim_A": 1,
                "dim_B": heads_num * batch_size,
                "dim_C": seq_len,
                "dim_D": head_dim,
                "transpose_order": "CD",
            },
        )
        subgraph.add_node(K_transpose)
        subgraph.connect(K_projected.name, K_transpose.name, 0, 0)

        K_transposed = Data(name=f"{self.name}_K_transposed")
        subgraph.add_node(K_transposed)
        subgraph.connect(K_transpose.name, K_transposed.name, 0, 0)

        # step 3b: Reshape K_transposed from 4D (1, heads_num*B, head_dim, n) to 3D (heads_num*B, head_dim, n)
        K_reshape = create_reshape_view(
            name=f"{self.name}_K_reshape",
            target_shape=(heads_num * batch_size, head_dim, seq_len),
        )
        subgraph.add_node(K_reshape)
        subgraph.connect(K_transposed.name, K_reshape.name, 0, 0)

        # Step 4: MatMul Q @ K^T -> (heads_num*B, n, n)
        QK_matmul = MatMul(
            name=f"{self.name}_QK_matmul",
            attrs={
                "dim_A": heads_num * batch_size,
                "in_channels": head_dim,
                "out_channels": seq_len,
                "batch_size": seq_len,
            },
        )
        subgraph.add_node(QK_matmul)
        subgraph.connect(Q_projected.name, QK_matmul.name, 0, 0)
        subgraph.connect(K_reshape.name, QK_matmul.name, 0, 1)

        # Create zero bias for QK matmul
        qk_bias_tensor = torch.zeros((1, seq_len), dtype=torch.bfloat16)
        qk_bias = Data(
            name=f"{self.name}_QK_bias",
            shape=(1, seq_len),
            dtype=DataType.BF16,
            payload=qk_bias_tensor,
        )
        subgraph.add_node(qk_bias)
        subgraph.connect(qk_bias.name, QK_matmul.name, 0, 2)

        QK_result = Data(name=f"{self.name}_QK_result")
        subgraph.add_node(QK_result)
        subgraph.connect(QK_matmul.name, QK_result.name, 0, 0)

        # Step 5: Multiply by scale factor
        QK_scaled = Multiply(
            name=f"{self.name}_QK_scale",
            attrs={
                "output_dtype": DataType.BF16,
                "bc_mode": 2,  # scalar mode
                "mult_or_div": 0,  # multiply
                "scalar": scale_factor,
            },
        )
        subgraph.add_node(QK_scaled)
        subgraph.connect(QK_result.name, QK_scaled.name, 0, 0)

        QK_scaled_result = Data(name=f"{self.name}_QK_scaled")
        subgraph.add_node(QK_scaled_result)
        subgraph.connect(QK_scaled.name, QK_scaled_result.name, 0, 0)

        # Step 6: Apply Softmax
        softmax_op = Softmax(
            name=f"{self.name}_softmax",
            attrs={
                "input_shape": (heads_num * batch_size, seq_len, seq_len),
                "axis": -1,
            },
        )
        subgraph.add_node(softmax_op)
        subgraph.connect(QK_scaled_result.name, softmax_op.name, 0, 0)

        attn_weights = Data(name=f"{self.name}_attn_weights")
        subgraph.add_node(attn_weights)
        subgraph.connect(softmax_op.name, attn_weights.name, 0, 0)

        # Step 7: MatMul attention @ V -> (heads_num*B, n, head_dim)
        attn_V_matmul = MatMul(
            name=f"{self.name}_attn_V_matmul",
            attrs={
                "dim_A": heads_num * batch_size,
                "in_channels": seq_len,
                "out_channels": head_dim,
                "batch_size": seq_len,
            },
        )
        subgraph.add_node(attn_V_matmul)
        subgraph.connect(attn_weights.name, attn_V_matmul.name, 0, 0)
        subgraph.connect(V_projected.name, attn_V_matmul.name, 0, 1)

        # Create zero bias for attn_V matmul
        attn_v_bias_tensor = torch.zeros((1, head_dim), dtype=torch.bfloat16)
        attn_v_bias = Data(
            name=f"{self.name}_attn_V_bias",
            shape=(1, head_dim),
            dtype=DataType.BF16,
            payload=attn_v_bias_tensor,
        )
        subgraph.add_node(attn_v_bias)
        subgraph.connect(attn_v_bias.name, attn_V_matmul.name, 0, 2)

        attn_output = Data(name=f"{self.name}_attn_output")
        subgraph.add_node(attn_output)
        subgraph.connect(attn_V_matmul.name, attn_output.name, 0, 0)

        # Step 8: Reshape from 3D (heads_num*B, n, head_dim) to 4D (heads_num, B, n, head_dim)
        reshape_4d = create_reshape_view(
            name=f"{self.name}_reshape_4d",
            target_shape=(heads_num, batch_size, seq_len, head_dim),
        )
        subgraph.add_node(reshape_4d)
        subgraph.connect(attn_output.name, reshape_4d.name, 0, 0)

        # Step 9: Transpose from 4D (heads_num, B, n, head_dim) to 4D (B, n, heads_num, head_dim)
        # Step 9a: Transpose from 4D (heads_num, B, n, head_dim) to 4D (B, heads_num, n, head_dim)
        transpose_swap_ab = Transpose(
            name=f"{self.name}_transpose_swap_ab",
            attrs={
                "dim_A": heads_num,
                "dim_B": batch_size,
                "dim_C": seq_len,
                "dim_D": head_dim,
                "transpose_order": "AB",
            },
        )
        subgraph.add_node(transpose_swap_ab)
        subgraph.connect(reshape_4d.name, transpose_swap_ab.name, 0, 0)

        transposed_swap_ab = Data(name=f"{self.name}_transposed_swap_ab")
        subgraph.add_node(transposed_swap_ab)
        subgraph.connect(transpose_swap_ab.name, transposed_swap_ab.name, 0, 0)

        # Step 9b: Transpose from 4D (B, heads_num, n, head_dim) to 4D (B, n, heads_num, head_dim)
        transpose_swap_bc = Transpose(
            name=f"{self.name}_transpose_swap_bc",
            attrs={
                "dim_A": batch_size,
                "dim_B": heads_num,
                "dim_C": seq_len,
                "dim_D": head_dim,
                "transpose_order": "BC",
            },
        )
        subgraph.add_node(transpose_swap_bc)
        subgraph.connect(transposed_swap_ab.name, transpose_swap_bc.name, 0, 0)

        transposed_4d = Data(name=f"{self.name}_transposed_4d")
        subgraph.add_node(transposed_4d)
        subgraph.connect(transpose_swap_bc.name, transposed_4d.name, 0, 0)

        # Step 10: Reshape to 3D (B, n, heads_num*head_dim)
        reshape_3d_out = create_reshape_view(
            name=f"{self.name}_reshape_3d_out",
            target_shape=(batch_size, seq_len, inner_dim),
        )
        subgraph.add_node(reshape_3d_out)
        subgraph.connect(transposed_4d.name, reshape_3d_out.name, 0, 0)

        # Step 11: MatMul with output projection -> 3D (B, n, embed_dim)
        out_matmul = MatMul(
            name=f"{self.name}_out_proj",
            attrs={
                "in_channels": inner_dim,
                "out_channels": embed_dim,
                "batch_size": seq_len,
            },
        )
        subgraph.add_node(out_matmul)
        subgraph.connect(reshape_3d_out.name, out_matmul.name, 0, 0)

        # output_2d = Data(name=f"{self.name}_output_2d")
        # subgraph.add_node(output_2d)
        # subgraph.connect(out_matmul.name, output_2d.name, 0, 0)

        # # Step 14: Reshape to final output 3D (B, n, embed_dim)
        # final_reshape = create_reshape_view(
        #     name=f"{self.name}_final_reshape",
        #     target_shape=(batch_size, seq_len, embed_dim),
        # )
        # subgraph.add_node(final_reshape)
        # subgraph.connect(output_2d.name, final_reshape.name, 0, 0)

        input_mapping = {
            0: [(input_reshape_2d.name, 0)],
            1: [(qkv_projects["Q"].name, 1)],
            2: [(qkv_projects["K"].name, 1)],
            3: [(qkv_projects["V"].name, 1)],
            4: [(out_matmul.name, 1)],
            5: [(qkv_projects["Q"].name, 2)],
            6: [(qkv_projects["K"].name, 2)],
            7: [(qkv_projects["V"].name, 2)],
            8: [(out_matmul.name, 2)],
        }

        return {
            "subgraph": subgraph,
            "input_mapping": input_mapping,
            "output_mapping": {0: (out_matmul.name, 0)},
        }
