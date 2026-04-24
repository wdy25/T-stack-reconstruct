# Program & Compiler User Guide

The `program` module provides the interface for users to describe their computational workloads.

## Core Concepts

### Computational Graph
*   **Nodes**: Represent operations (e.g., Conv, Add, MatMul).
*   **Edges**: Represent data dependencies.

### Configuration Generation
The compiler takes the high-level graph and maps it to the hardware resources, generating a configuration file (or object).

## API Reference

### Creating a Graph
TODO: Describe `graph` class and methods.

### defining Operations
TODO: List available primitives in `operations/` (e.g., `operations.add`, `operations.conv`).

### Generating Configs
TODO: Describe `config_gen`.
