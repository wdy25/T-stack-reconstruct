# Compiler 用户指南

Compiler 模块负责将通过 `HardwareGraph` 定义的高级计算图转换为硬件可执行的二进制指令和配置文件。编译过程包含内存分配、操作调度、图检查、代码生成和配置生成等步骤。

## 编译流程概览

编译一个 `HardwareGraph` 通常遵循以下顺序：

1.  **内存分配 (Memory Allocation)**: 为图中的数据节点分配具体的硬件内存地址。
2.  **操作调度 (Operation Scheduling)**: 确定每个核心上的指令执行顺序及依赖关系。
3.  **图检查 (Graph Check)**: 验证硬件图的完整性和正确性。
4.  **代码生成 (Code Generation)**: 将操作和依赖转换为机器码。
5.  **配置生成 (Config Generation)**: 输出最终的内存初始化文件和控制配置文件。

## 详细步骤

假设您已经构建好了一个 `HardwareGraph` 对象 `hwg`，并且已经完成了核心映射（使用 `set_core_id_for_nodes`）和通信节点生成（使用 `gen_communication_ops`）。

### 1. 内存分配 (Memory Allocator)

`MemoryAllocator` 负责计算所有 `Data` 节点的生命周期，并在给定的内存空间内为其分配地址。

```python
from core.compiler.memory_allocator import MemoryAllocator

# 1. 生成原语的额外参数。额外参数将变为数据节点
hwg.gen_para_nodes()

# 2. 初始化分配器
allocator = MemoryAllocator(hwg)

# 3. 执行分配
# mem_per_core: 每个核心的可用内存大小 (单位: 元素个数)
# reserved_space:保留地址空间 (主要留给原语存储)
# non_overwritable_patterns: 不允许覆盖的数据名称正则列表 (如权重、输入)
# incremental: 是否使用增量分配策略
allocator.allocate_memory(
    mem_per_core=16384, 
    reserved_space=256, 
    non_overwritable_patterns=['.*weight', '.*bias', '.*para.*', 'input'], 
    incremental=True
)

# (可选) 可视化内存生命周期，生成png图片
allocator.visualize_lifecycle("output/memory_lifecycle")
```

### 2. 操作调度 (Operation Scheduler)

`OperationScheduler` 决定每个核心执行操作的线性顺序，并计算每条指令的依赖掩码（Dependency Mask）。

```python
from core.compiler.operation_scheduler import OperationScheduler

# 1. 初始化调度器
op_sch = OperationScheduler(hwg)

# 2. 构建操作列表
# try_parallel: 尽可能利用多指令并行
op_lists = op_sch.build_core_op_lists(try_parallel=True)

# 3. 构建依赖关系
# max_backtrack: 依赖掩码的最大回溯长度 (例如 8 表示可以依赖前 8 条指令)
deps = op_sch.build_deps_for_ops(max_backtrack=8)

# op_lists 是一个字典: {core_id: [node_id1, node_id2, ...]}
# deps 有着相同的结构，存储对应的依赖掩码
```

### 3. 图检查 (Check Graph)

在生成代码之前，使用 `CheckGraph` 确保硬件图满足所有约束条件。

```python
from core.compiler.check_graph import CheckGraph

checker = CheckGraph()
is_valid = checker.check(hwg)
if not is_valid:
    raise ValueError("Hardware Graph validation failed.")
```

检查项包括：
*   所有节点是否都有 `core_id`。
*   数据节点是否有合法地址。
*   输入数据是否包含 `payload`。
*   是否有孤立节点。
TODO: 添加更多检查项

### 4. 代码生成 (Code Generator)

`CodeGenerator` 结合硬件图、操作列表和依赖关系，生成最终的机器指令列表。

```python
from core.compiler.code_generator import CodeGenerator

# 1. 初始化代码生成器
code_gen = CodeGenerator(hwg, op_lists, deps)

# 2. 生成代码
# auto_stop: 是否在末尾自动添加停止指令
codes = code_gen.generate_code(auto_stop=True)

# codes 是一个字典: {core_id: [instruction_intbv, ...]}
```

### 5. 配置生成 (Config Generator)

最后，`ConfigGenerator` 将生成的代码和初始内存数据导出为文件。这些文件可以被 Emulator 加载，或用于实际芯片的配置。

```python
from core.compiler.config_gen import ConfigGenerator

# 1. 配置生成器设置
settings = {
    "simple": False,
    "continuous": True,
    # 其他硬件特定配置...
}

# 2. 初始化生成器
config_gen = ConfigGenerator(
    hwg, 
    codes, 
    output_dir="output/config_files", 
    settings=settings
)

# 3. 生成标准配置文件 (ctrl 和 mem 文件)
config_gen.generate_all_configs()

# (可选) 生成 LVDS C 语言配置 (用于多芯片互联场景)
# config_gen.generate_lvds_c_configs()
```