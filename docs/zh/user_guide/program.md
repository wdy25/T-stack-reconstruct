# 编程指南

`ir` 模块为用户提供了描述计算工作负载的接口。

## IR

### 计算图 (Computational Graph)
所有工作负载都用计算图描述

#### **节点 (Nodes)**
* Operation: 代表算子操作 (例如 Conv, Add, MatMul)。
    * 大多数算子都代表计算过程
    * Communication Operation: 通信算子，用于核间数据传输
    * Control Operation: 控制算子，用于控制核内的控制流
* Data: 代表张量数据
    * 普通的数据节点代表张量以及存储中的数据块
    * ViewData用于对Data节点变形/索引。数据变形和索引不会改变数据在Memory中的存储方式，因此有一定限制
    * ConcatData用于拼接数据，将两份或多份Data视为一份。连接到同一个ConcatData节点的数据在Memory上会被连续存储

#### **边 (Edges)**
* 数据边：连接数据与算子或数据与ViewData/ConcatData。代表数据依赖关系
* 控制边：连接算子与算子。代表算子运行顺序

### 硬件图 HardwareGraph
在计算图的基础上，加上各种硬件信息，包括但不限于
* 算子/数据节点的核坐标
* 数据节点对应的存储信息

### 原语 Primitives
芯片/架构底层的指令，普通用户不需要了解，请参考Developer Guide

### API参考

#### 创建图 (Creating a Graph)

在使用 IR 描述计算任务时，首先需要创建一个 `Graph` 对象。

```python
from core.ir.graph import Graph
g = Graph()
```

#### Graph 类方法

*   **`add_node(node: Any) -> NodeId`**
    *   向图中添加一个节点。节点可以是 `Data`、`ViewData`、`ConcatData` 或 `Operation` 的实例。
    *   返回添加节点的唯一标识符（即节点的 `name`）。
    *   *Tips: 节点的 `name` 在图中必须是唯一的。*

*   **`connect(src: NodeId, dst: NodeId, src_port: Optional[int] = None, dst_port: Optional[int] = None)`**
    *   在两个节点之间建立连接。
    *   `src`: 源节点的名称。
    *   `dst`: 目标节点的名称。
    *   `src_port` / `dst_port`: 指定源和目标的端口号。
        *   通常，`Data` 节点连接到 `Operation` 时，需要指定 `Operation` 的输入端口 (`dst_port`)。
        *   `Operation` 输出到 `Data` 节点时，需要指定 `Operation` 的输出端口 (`src_port`)。
        *   如果不指定端口，系统会尝试自动分配下一个可用的端口（主要用于简单的顺序连接）。

*   **`infer()`**
    *   执行静态形状和类型推断。
    *   该方法会按照拓扑顺序遍历图，调用每个算子的 `infer` 方法，推导出所有中间 Data 节点的 `shape` 和 `dtype`。
    *   通常在图构建完成后，编译之前调用。

```python
g.infer()
```

*   **`to_prim()`**
    *   将图中的高级算子（High-level Operations）转换为底层原语（Primitives）。
    *   编译器后端通常只支持原语。如果定义的图中包含复合算子（如标准的 `Conv` 可能被展开为 `DeepConv` 或其他硬件原语），需要调用此方法进行转换。
    *   转换是原地的（in-place），会修改当前的图结构。

```python
g.to_prim()
```

*   **`visualize(output_path: str = "graph", format: str = "png", vertical: bool = True)`**
    *   生成图的可视化文件。需要安装 `graphviz`。
    *   `output_path`: 输出文件的路径（不含后缀）。
    *   `format`: 输出格式 (png, pdf, svg 等)。
    *   `vertical`: 是否使用垂直布局 (Top-Down)。

```python
g.visualize("my_graph_vis", format="pdf")
```

#### **Data**
基础数据节点，包含数据的形状、类型和可选的初始值。

```python
from core.ir.data import Data, DataType
import torch

# 定义一个 INT8 输入数据
input_data = Data(
    name="input", 
    shape=(1, 32, 32, 32), 
    dtype=DataType.INT8, 
    payload=torch.zeros((1, 32, 32, 32), dtype=torch.int8)
)
g.add_node(input_data)
```

关键参数：
*   `name` (str): 数据节点名称。
*   `shape` (Tuple[int, ...]): 张量形状。
*   `dtype` (DataType): 数据类型，支持 `INT8`, `BF16`, `INT32`, `SPIKE`。
*   `payload` (Optional[Union[torch.Tensor, str]]): 初始数据常量。如果是输入数据或权重，通常需要提供。如果是中间结果，可以为 `None`。

#### **ViewData**
视图节点不存储实际数据，而是提供了另一种访问现有数据的方式（例如 reshape 或 slice）。它不产生数据拷贝，直接复用源数据的内存。

```python
from core.ir.data import ViewData

# Reshape 视图
view_node = ViewData(
    name="reshaped_view",
    view_type="reshape",
    target_shape=(1, 32, 1024)
)
g.add_node(view_node)
g.connect("input", "reshaped_view") # 将源数据连接到视图
```

*   `view_type`: 支持 `"reshape"`, `"slice"`, `"reshape_and_slice"`。
*   `target_shape`: 用于 reshape 模式的目标形状。
*   `slice_dimension_index`, `slice_start`, `slice_end`: 用于 slice 模式的切片参数。

#### **ConcatData**
用于将多个数据块在逻辑上拼接成一个连续的内存块。
TODO: 补充

#### Operations

所有算子都位于 `core.ir.operations` 包中。算子通常需要指定输入输出端口以及特定的属性 (`attrs`)。

通用参数：
*   `name` (str): 算子名称。
*   `attrs` (Dict[str, Any]): 算子的特定属性字典。

#### 常用算子列表

*   **Conv (Convolution)**: 标准卷积操作。
    *   `attrs`: `kernel_size`, `stride`, `padding`, `dilation`, `in_channels`, `out_channels`。
*   **DeepConv**: 深度卷积（通常指特定硬件优化的卷积实现）。
*   **Add**: 元素级加法，支持广播。
    *   `attrs`: `output_dtype`, `bc_mode` (广播模式), `add_or_sub` (0=加, 1=减)。
*   **Multiply**: 元素级乘法。
*   **MatMul**: 矩阵乘法。
    *   `attrs`: `dim_A`, `in_channels`, `out_channels`, `batch_size`。
*   **MaxPooling** / **MeanPooling**: 池化操作。
    *   `attrs`: `kernel_size`, `stride`.
*   **Relu** / **NonLinear**: 激活函数。
*   **Stop**: 停止符，用于控制流。

#### 算子定义示例

**卷积层示例:**
```python
from core.ir.operations.conv import Conv

conv_op = Conv(
    name="conv1", 
    attrs={
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (1, 1, 1, 1),
        "in_channels": 32,
        "out_channels": 64
    }
)
g.add_node(conv_op)

# 连接: Input(Data) -> (port 0) Conv (port 0) -> Output(Data)
# 还需要连接权重和偏置到 Conv 的 port 1 和 port 2
g.connect("input_data", "conv1", dst_port=0)
g.connect("weight_data", "conv1", dst_port=1)
g.connect("bias_data", "conv1", dst_port=2) 
g.connect("conv1", "output_data", src_port=0)
```

**加法层示例:**
```python
from core.ir.operations.add import Add
from core.ir.data import DataType

add_op = Add(
    name="add1",
    attrs={
        "output_dtype": DataType.BF16,
        "bc_mode": 0, # 0: No broadcast
        "add_or_sub": 0
    }

### 控制流连接 (Control Flow Connection)

除了数据依赖，有时还需要强制规定算子的执行顺序（例如确保初始化完成后再执行计算）。

*   **`connect_control(src: NodeId, dst: NodeId)`**
    *   在 `src` 和 `dst` 之间添加一条控制边，表示 `src` 必须在 `dst` 之前执行。
    *   控制边只能连接 `Operation` (或 `ControlOp` / `CommunicationOp`) 节点，不能连接 `Data` 节点。

```python
# 示例：强制 stop_op 在 add_op 之后执行
g.connect_control("add_op", "stop_op")
```


### 硬件图 (HardwareGraph)

`HardwareGraph` 是 `Graph` 的子类，增加了硬件相关的信息，如核心映射 (Core Mapping)、内存地址分配 (Memory Allocation) 和跨核通信 (Communication)。

```python
from core.ir.hardware_graph import HardwareGraph

# 从现有计算图创建硬件图
hwg = HardwareGraph(g)
```

#### 核映射 (Core Mapping)

*   **`set_core_id(node_id: NodeId, core_id: Tuple[int, int])`**
    *   将单个节点分配到指定的计算核。`core_id` 通常是二维坐标 `(x, y)`。
*   **`set_core_id_for_nodes(node_ids: Iterable[NodeId], core_id: Tuple[int, int])`**
    *   将一组节点批量分配到指定的计算核。
*   **`split(split_edges: List[Tuple[NodeId, NodeId]]) -> List[List[NodeId]]`**
    *   根据指定的边将图分割成多个子图（连通分量）。
    *   常用于自动划分多核任务。返回节点列表的列表。

#### 硬件信息生成 (Hardware Feature Generation)

*   **`gen_memref_for_all_data()`**
    *   为所有 Data 节点生成 `MemBlock` 对象。
    *   这是内存分配的前提步骤，确定每个数据块的大小和对齐需求。

*   **`gen_communication_ops()`**
    *   分析跨核的数据依赖。
    *   如果在 Core A 产生的数据被 Core B 使用，此方法会自动在 Core A 插入 `Send` 算子，在 Core B 插入 `Recv` 算子，并处理数据搬运逻辑。

*   **`gen_para_nodes()`**
    *   为算子生成参数节点（Parameter Nodes）。
    *   某些硬件算子需要特定的配置参数存储在内存中，此方法会自动生成这些参数数据节点并连接到算子。

*   **`visualize(...)`**
    *   与 `Graph` 的 visualize 类似，但在图中会额外显示节点所属的 **Core ID** 以及内存配置信息，不同类型的节点（Data, View, Op, Control）会有不同的形状和颜色标识。

```python
# 典型的多核编译流程片段
hwg = HardwareGraph(g)
# 1. 划分核心
nodes_part1 = [...] 
nodes_part2 = [...]
hwg.set_core_id_for_nodes(nodes_part1, (0, 0))
hwg.set_core_id_for_nodes(nodes_part2, (0, 1))

# 2. 生成通信
hwg.gen_communication_ops()

# 3. 准备内存分配
hwg.gen_memref_for_all_data()

# 4. 可视化检查
hwg.visualize("hardware_mapping", vertical=True)
```



