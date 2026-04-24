# 开发者指南：架构概览

本节解释 T-Stack 项目的内部结构。

## 高层架构

本项目结构分为三个主要层级：

1.  **编程/编译 (Program/Compiler)**: 处理图构建、优化和映射。位于 `core_components/` 和 `operations/`。
2.  **仿真 (Simulation - Emulator)**: 提供硬件的行为模型。
3.  **分析 (Analysis - Analyser)**: 提供性能建模。

## 目录结构

*   `archive/`: 遗留代码和备份。
*   `convert/`: 文件格式转换工具 (txt <-> mem/coe)。
*   `core_components/`: 编译器/图引擎的核心逻辑 (`graph.py`, `code_generator.py` 等)。
*   `operations/`: 编译器使用的高级操作定义。
*   `prims/`: 低级原语实现 (可能用于 emulator/analyser)。
*   `tests/`: 集成测试和单元测试。

## 开发指南

*   **[添加新算子与原语](add_new_op_and_prim.md)**: 详细介绍如何扩展 T-Stack 的指令集，包括开发 Emulator/Analyser 的原语以及 Compiler 的算子。

