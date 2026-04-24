# Developer Guide: Architecture Overview

This section explains the internals of the T-Stack project.

## High-Level Architecture

The project is structured into three main layers:

1.  **Frontend (Program/Compiler)**: Handles graph construction, optimization, and mapping. Located in `core_components/` and `operations/`.
2.  **Simulation (Emulator)**: Provides a behavioral model of the hardware.
3.  **Analysis (Analyser)**: Provides performance modeling.

## Directory Structure

*   `archive/`: Legacy code and backups.
*   `convert/`: Tools for file format conversion (txt <-> mem/coe).
*   `core_components/`: Core logic for the compiler/graph engine (`graph.py`, `code_generator.py`, etc.).
*   `operations/`: High-level operation definitions usable by the compiler.
*   `prims/`: Lower-level primitive implementations (likely for emulator/analyser).
*   `tests/`: Integration and unit tests.
