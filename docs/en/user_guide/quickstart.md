# Quick Start

This section will guide you through running a simple example using the T-Stack toolchain.

## Prerequisites
*   Python 3.x installed.
*   Dependencies listed in `requirements.txt` installed.

## Running a Basic Example (Add Operation)

You can find a basic example of an addition operation in `tests/emulator/prims/add_test.py`.

### 1. Define the Graph
(Code snippet placeholder showing how to define an Add operation using `program`)

### 2. Generate Configuration
(Code snippet placeholder showing `config_gen`)

### 3. Run Simulation
(Code snippet placeholder showing `emulator.run`)

## Running a Network (ResNet9)

For a more complex example, refer to `tests/program/networks/resnet9/resnet9_test.py`.

```bash
# Example command to run the test
python tests/program/networks/resnet9/resnet9_test.py
```

## Performance Analysis
To analyze the performance of an operation, refer to `tests/analyser/add_test.py`.
