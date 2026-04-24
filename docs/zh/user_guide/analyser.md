# Analyser 用户指南

Analyser 是一个性能分析工具，用于在软件层面预估模型的执行延迟、资源利用率和带宽消耗。与 Emulator 不同，Analyser 更侧重于性能指标的统计而非功能的精确验证。

## 特性

*   **延迟预估 (Latency estimation)**: 基于事件驱动的仿真引擎，估算程序在目标硬件上的执行时间。
*   **资源利用率 (Resource utilization)**: 统计矩阵单元 (Matrix)、向量单元 (Vector) 和内存 (Memory) 的使用率及功耗。
*   **Trace 生成**: 生成可视化 Trace 文件，展示各硬件组件的任务执行时间轴。
*   **存储挤占**: 仿真引擎会根据存储带宽资源与事件的存储带宽需求来估算事件的运行时间

## 用法

Analyser 的使用流程通常包括：初始化芯片阵列 -> 部署图 -> 运行分析引擎 -> 获取报告。

### 1. 初始化与部署 (Setup & Deploy)

首先需要创建 `ChipArray` 对象并加载硬件配置。接着，将编译生成的硬件图部署到阵列上。

```python
from core.simulator.analyser.chip_array import ChipArray
from basics.utils import load_config_from_yaml

# 1. 加载配置文件
config = load_config_from_yaml("core/simulator/configs/basic_config.yaml")

# 2. 初始化芯片阵列
# chip_array_size: 芯片网格大小 (例如 (1, 1) 表示单芯片)
# core_array_size: 每个芯片内的核心网格大小 (例如 (2, 2) 表示每芯片4核)
chip_array = ChipArray(chip_array_size=(1, 1), core_array_size=(2, 2), config=config)

# 3. 部署程序
# hardware_graph, op_lists, deps 均来自编译器的输出
chip_array.deploy(hardware_graph, op_lists, deps)
```

### 2. 运行分析 (Run Engine)

创建 `Engine` 实例并运行仿真。`Engine` 采用离散事件仿真（Discrete Event Simulation）机制，快速跳过空闲时间，比周期精确仿真更快。

```python
from core.simulator.analyser.engine import Engine

# 初始化引擎
engine = Engine(chip_array, config)

# 运行仿真
# 仿真会自动进行直到所有事件处理完毕
engine.run()
```

### 3. 获取报告 (Reports)

#### 资源利用率报告

使用 `printUtilizations()` 方法打印每个核心的详细资源使用情况。

```python
engine.printUtilizations()
```

输出示例解读：
*   **duration**: 核心运行的总周期数。
*   **matrix/vector/memory**: 各单元的利用率统计。
    *   `max_resource_utilization`: 峰值利用率。
    *   `average_resource_utilization`: 平均利用率。
    *   `energy`: 估算的能耗（纳焦 nJ）。
*   **Total computation/energy**: 总计算量和总能耗。

#### Trace 可视化

使用 `getTrace()` 方法生成 Trace 文件（Protobuf 格式）。该文件与 Perfetto 兼容，可用于可视化分析执行流水线。

```python
# 生成 trace 文件
# path: 输出文件的路径 (通常以 .pb 结尾)
# 注意：生成的 protobuf 文件需要使用支持该格式的查看器打开 (如 https://ui.perfetto.dev/)
engine.getTrace(path="temp/trace.pb")
```

可以使用 https://ui.perfetto.dev/ 打开pb文件

在 Trace 视图中，你可以看到：
*   每个核心 (Core) 作为独立的进程 (Process)。
*   矩阵单元、向量单元、传输单元等作为线程 (Thread)。
*   各个操作的时间跨度 (Slices)，清晰展示并行度和流水线气泡。

## 完整示例

请参考 `tests/analyser/prims/add_test.py` 获取完整的代码示例，该示例展示了如何构建一个简单的加法图并进行性能分析。
