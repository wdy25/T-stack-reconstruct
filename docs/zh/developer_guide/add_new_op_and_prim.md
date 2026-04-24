# 添加新算子与原语指南

在 T-Stack 中，**算子 (Operation)** 是构建计算图（HardwareGraph）并被编译器处理的基本单元，而 **原语 (Primitive)** 是硬件执行（Emulator）或性能分析（Analyser）的基本单元。

通常，开发一个新功能（如新的算术运算）需要同时实现这一对组件。

---

## 1. 原语开发 (Primitive Development)

原语定义了操作在模拟器中的行为（功能）和在分析器中的行为（性能）。原语代码位于 `core/ir/prims/` 目录下。

所有原语类必须继承自 `Primitive` 类。

### 1.1 核心结构

参考 `core/ir/prims/add.py`，一个原语类通常包含：

*   **`__init__`**: 初始化参数。
*   **`setPIC`**: 定义指令编码 (Binary Encoding)。
*   **`execute`**: Emulator 执行逻辑。
*   **`generate_events`**: Analyser 事件生成逻辑。

### 1.2 `execute` 函数 (功能仿真)

`execute` 函数接收一个 `EmuCore` 对象，允许你访问模拟的内存和寄存器。你需要模拟硬件的行为：由于是 Cycle-Accurate 或功能级仿真，通常直接使用 PyTorch/NumPy 完成计算。

```python
from core.simulator.emulator.core import Core as EmuCore

def execute(self, core: EmuCore):
    # 1. 准备参数
    # 例如：计算两个输入张量的长度
    input1_len = ... 
    
    # 2. 读取内存 (Memory Read)
    # core.memory 是模拟内存对象。切片操作返回内存数据。
    # 使用 .view() 和 .reshape() 恢复 Tensor 形状
    raw_data = core.memory[self.x_in_1_addr : self.x_in_1_addr + input1_len]
    input1_tensor = raw_data.view(torch.bfloat16).reshape(...)
    
    # 3. 执行计算 (Logic)
    # 使用 PyTorch 进行实际的数学运算
    output_tensor = input1_tensor + input2_tensor
    
    # 4. 写回内存 (Memory Write)
    # 将结果写回指定的输出地址
    core.memory.writeTensor(self.y_out_addr, output_tensor)
```

### 1.3 `generate_events` 函数 (性能分析)

`generate_events` 函数接收一个 `AnalyCore` 对象，返回一系列 `Event` 对象（如计算事件、内存读写事件）。Analyser 引擎利用这些事件来模拟延迟和功耗。

```python
from core.simulator.analyser.core import Core as AnalyCore
from core.simulator.analyser.event import ComputeEvent, MemoryEvent, EventType

def generate_events(self, core: AnalyCore):
    events = []
    
    # 1. 计算计算量 (Computation)
    # 根据输入形状和并行度计算所需的周期数或操作数
    computation_ops = ... 
    cycles = ...
    
    # 创建计算事件
    compute_event = ComputeEvent(
        name="Add_Compute",
        parent=core.vector.full_name, # 指定执行单元 (Matrix/Vector)
        compute_type=EventType.VECTOR,
        computation=computation_ops,
        # ... 其他能量和速率参数 ...
    )
    events.append(compute_event)
    
    # 2. 计算访存量 (Memory Access)
    # 判断地址属于哪个层级的内存 (L0/L1)
    if self.x_in_1_addr < core.config["core"]["L0_memory_capacity"] ...:
        hierarchy = 0
    else:
        hierarchy = 1
        
    # 创建内存读事件
    mem_read_event = MemoryEvent(
        name="Add_Read_Input1",
        parent=core.memory[hierarchy].full_name,
        memory_type=EventType.READ,
        volume=input_volume_bytes,
        hierarchy=hierarchy,
        # ...
    )
    events.append(mem_read_event)
    
    # 同样地，创建内存写事件...
    
    return events
```

---

## 2. 算子开发 (Operation Development)

算子是用于构建图的高级接口。算子代码位于 `core/ir/operations/` 目录下，需继承自 `Operation` 类。

根据算子与硬件原语的关系，开发分为三种场景：

### 2.1 场景 A：低级算子 (Low-level Operator)

**适用情况**：算子与硬件原语一一对应（例如 `Add`, `Lif`, `MatMul`）。

**开发要点**：
*   **`self.primitive = True`**: 在 `__init__` 中标记此算子为原子操作，不需要展开。
*   **`infer`**: 实现形状推断和数据类型检查。
*   **`gen_prim`**: 用于代码生成。接收 `Data` 对象和依赖，实例化对应的 Primitive 类并返回其 `PIC` (Packed Instruction Code)。
*   **`build_prim`**: 用于仿真器加载。接收 `Data` 对象和依赖，返回对应的 Primitive 对象实例。

示例参考：`core/ir/operations/lif.py`

```python
class Lif(Operation):
    def __init__(self, name, attrs):
        super().__init__(name, attrs)
        self.primitive = True  # 关键标记
        
    def infer(self, inputs):
        # 校验输入数量、类型
        # 返回 [(输出形状, 输出类型), ...]
        return [(inputs[0].shape, DataType.BF16)]
        
    def gen_prim(self, inputs, outputs, deps):
        # 提取地址
        vin_addr = inputs[0].memref.addr
        # ...
        
        # 实例化原语并获取二进制编码
        prim = PrimLif(deps=deps, Vin_addr=vin_addr, ...)
        return prim.PIC
        
    def build_prim(self, inputs, outputs, deps):
        # 返回用于仿真的原语对象
        return PrimLif(deps=deps, ...)
```

### 2.2 场景 B：可展开的高级算子 (High-level/Composite Operator)

**适用情况**：算子逻辑较复杂，对应多个硬件原语，或者包含隐含的数据节点（如权重）。例如 `Conv` (卷积) 可以展开为 `DeepConv` (计算) + `Weights` (数据) + `Bias` (数据)。

**开发要点**：
*   **`self.primitive = False`**: 标记此算子需要展开。
*   **`to_prim`**: 核心方法。返回一个包含子图的字典。编译器会在后续处理中用这个子图替换该节点。

示例参考：`core/ir/operations/conv.py`

```python
class Conv(Operation):
    def __init__(self, name, attrs):
        super().__init__(name, attrs)
        self.primitive = False  # 关键标记
        
    def to_prim(self):
        subgraph = Graph()
        
        # 1. 创建子图内部节点
        # 通常包含一个更底层的 Operator (如 DeepConv)
        real_op = DeepConv(self.name, self.attrs)
        subgraph.add_node(real_op)
        
        # 2. (可选) 创建隐含的参数数据节点
        w_data = Data(f"{self.name}.weight", ...)
        subgraph.add_node(w_data)
        
        # 3. 建立内部连接
        # 连接权重到计算节点
        subgraph.connect(w_data.name, self.name, 0, 1) # 权重连到 DeepConv 的端口 1
        
        # 4. 返回替换信息
        return {
            "subgraph": subgraph,
            # 输入映射: {父节点的端口: (子图内节点的名称, 子图内节点的端口)}
            "input_mapping": {0: (self.name, 0)},
            # 输出映射
            "output_mapping": {0: (self.name, 0)}
        }
```

### 2.3 场景 C：复杂子图构建 (Complex Subgraph Factory)

**适用情况**：算子非常复杂，难以通过单一的 `Operation` 类及其 `to_prim` 方法优雅地描述（例如多输入多输出的复杂 Block），或者用户希望拥有更大的灵活性。

**建议做法**：
不要继承 `Operation` 类。直接编写一个 Python 函数（Factory Function），在其中实例化多个算子并添加到传入的 `Graph` 中。

```python
def add_complex_attention_block(graph: Graph, input_node_name: str, config: dict) -> str:
    """
    向图中添加一个复杂的注意力块，并返回输出节点的名称。
    """
    # 1. 创建所有需要的算子
    q_proj = MatMul(f"{input_node_name}_q", ...)
    k_proj = MatMul(f"{input_node_name}_k", ...)
    softmax = Softmax(...)
    
    # 2. 将节点加入图
    graph.add_node(q_proj)
    graph.add_node(k_proj)
    graph.add_node(softmax)
    
    # 3. 创建中间数据节点并连接
    q_out = Data(f"{input_node_name}_q_out")
    graph.add_node(q_out)
    
    graph.connect(input_node_name, q_proj.name) # 连接输入
    graph.connect(q_proj.name, q_out.name)
    # ... 连接其他逻辑 ...
    
    # 4. 返回最终输出节点的名字，供后续连接使用
    return final_output_node_name
```
