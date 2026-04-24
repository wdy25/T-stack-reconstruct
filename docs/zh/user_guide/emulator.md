# Emulator 用户指南

Emulator 允许您在软件仿真环境中运行生成的硬件图配置。这对于在实际硬件上运行之前进行调试和功能验证非常有用。它完全用 Python 编写，可以直接与编译器的中间表示（IR）进行交互。

## 特性

* **非精确仿真**: Emulator的仿真介于“运行计算图”和“模拟芯片”之间。基于简化的芯片模型进行仿真。不考虑具体运行时钟数，不考虑原语并行。Emulator与实际芯片的功能一致性需要compiler支持（即由compiler保证生成的配置在并行和非并行模式下行为一致）。
* **直接集成**: 直接使用编译器的 `HardwareGraph` 和调度结果进行部署，无需中间文件转换。
* **状态检查**: 可以随时检查内存和核心状态。
* **数据验证**: 方便地提取仿真结果并与预期输出（如 PyTorch 计算结果）进行对比。

## 用法

Emulator 的典型工作流包含三个步骤：初始化与部署、运行仿真、结果验证。

### 1. 加载配置 (Deploy)

Emulator 不直接读取二进制或文本配置文件，而是通过 `deploy` 方法直接接收编译器的输出对象。你需要准备好以下对象：

*   **硬件图 (HardwareGraph)**: 包含所有数据节点和计算节点的图结构，且已完成内存分配。
*   **操作列表 (op_lists)**: 每个核心按顺序执行的操作节点 ID 列表。
*   **依赖关系 (deps)**: 每个操作对应的依赖位掩码（dependency bitmask）。

这些通过 `OperationScheduler` 和 `CodeGenerator` 等组件生成。

```python
from core.simulator.emulator.chip import Chip as EmulatorArray
from basics.utils import load_config_from_yaml

# 1. 初始化仿真器阵列
# config: 加载硬件配置文件（如内存大小、时序等）
# array_size: 核心阵列的大小，例如 (1, 1) 表示单核
config = load_config_from_yaml("core/simulator/configs/basic_config.yaml")
core_array = EmulatorArray(config=config, array_size=(1, 1))

# 2. 部署程序
# hardware_graph: 完成编译的硬件图
# op_lists: 调度器生成的每个核心的操作序列
# deps: 调度器生成的依赖关系
core_array.deploy(hardware_graph, op_lists, deps)
```

在部署过程中，Emulator 会将 `HardwareGraph` 中的初始数据（如权重、偏置、输入数据）自动写入到对应核心的模拟内存中。

### 2. 运行仿真 (Run)

使用 `run()` 方法启动仿真。该方法会模拟芯片运行，直到所有核心完成其操作序列。

```python
# 运行仿真
# from_start=True 表示从头开始运行，会重置核心状态
core_array.run(from_start=True)
```

Emulator 会打印每个核心的完成状态。如果所有核心都执行完了 `Stop` 指令之前的操作，则视为成功。

### 3. 验证输出 (Verify)

仿真完成后，你可以从模拟内存中读取结果数据。为了方便定位数据，可以使用 `output_if` 模块中的 `get_output_list` 辅助函数。

```python
from core.simulator.emulator.output_if import get_output_list
import torch

# 1. 确定要获取的输出节点名称
output_node_names = ["output_tensor_name"]

# 2. 生成输出接口描述
# 这会根据 HardwareGraph 自动查找数据的核心位置、内存地址、形状和数据类型
output_descriptors = get_output_list(hardware_graph, output_node_names)

# 3. 获取数据
# 返回一个字典，键为节点 ID 或名称，值为 PyTorch Tensor
outputs = core_array.get_outputs(output_descriptors)

# 4. 与预期结果比较
true_output = ... # 计算出的预期结果
print(torch.mean(torch.abs(outputs["output_tensor_name"] - true_output)))
```

`get_outputs` 方法会自动处理跨核心读取和数据类型转换（例如将 BF16 转换回 PyTorch 的 Tensor）。

## 完整示例

请参考 `tests/emulator/prims/add_test.py` 获取完整的代码示例，该示例展示了从构建图到仿真的全过程。
