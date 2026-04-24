# 用户指南概览

本指南将介绍如何使用 T-Stack 工具链。典型的工作流包括：

1.  **定义计算图**: 使用 `program` 模块定义操作和数据流。
2.  **编译**: 将计算图转换为硬件配置。
3.  **执行/仿真**:
    *   在 **Emulator** 上运行以验证正确性。
    *   在 **Analyser** 上运行以评估性能指标。
    *   在 **Hardware** 上运行 (如可用)。

## 章节索引

*   **[快速开始 (Quick Start)](quickstart.md)**: 运行你的第一个示例 (Add 操作或 ResNet)。
*   **[Program](program.md)**: 定义计算图的操作和数据流。
*   **[Compiler](compiler.md)**: 详细讲解编译流程，包括内存分配、调度和代码生成。
*   **[Emulator](emulator.md)**: 如何加载配置和数据进入仿真器进行功能验证。
*   **[Analyser](analyser.md)**: 如何运行性能分析并解读报告。
