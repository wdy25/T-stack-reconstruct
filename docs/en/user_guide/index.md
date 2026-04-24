# User Guide Overview

This guide describes how to use the T-Stack toolchain. The typical workflow involves:

1.  **Defining the Computational Graph**: Using the `program` module to define operations and data flow.
2.  **Compilation**: Transforming the graph into a hardware configuration.
3.  **Execution/Simulation**:
    *   Running on the **Emulator** to verify correctness.
    *   Running on the **Analyser** to estimate performance metrics.
    *   Running on actual **Hardware** (if available).

## Sections

*   **[Quick Start](quickstart.md)**: Run your first example (Add operation or ResNet).
*   **[Program / Compiler](program.md)**: Detailed API for graph construction and configuration generation.
*   **[Emulator](emulator.md)**: How to load configurations and data into the emulator.
*   **[Analyser](analyser.md)**: How to interpret performance analysis results.
