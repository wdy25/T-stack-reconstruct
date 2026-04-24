# T-Stack 项目文档

欢迎查阅 T-Stack 项目文档。本项目包含三个核心组件：

1.  **Program (Compiler)**: 用于构建计算图并生成硬件配置的工具。
2.  **Emulator**: 用于验证配置和功能正确性的仿真环境。
3.  **Analyser**: 用于评估硬件执行性能的分析工具。

## 文档结构

文档分为两个主要部分：

### [用户指南 (User Guide)](user_guide/index.md)
面向希望使用该工具链构建模型、进行仿真和性能分析的用户。
- [快速开始 (Quick Start)](user_guide/quickstart.md)
- [Program & Compiler 指南](user_guide/program.md)
- [Emulator 指南](user_guide/emulator.md)
- [Analyser 指南](user_guide/analyser.md)

### [开发者指南 (Developer Guide)](developer_guide/index.md)
面向希望理解内部架构并扩展功能的贡献者和开发者。
- [架构概览](developer_guide/index.md)
- [Compiler (Program) 设计](developer_guide/compiler_design.md)
- [Emulator 设计](developer_guide/emulator_design.md)
- [Analyser 设计](developer_guide/analyser_design.md)
