# Emulator Design

The emulator attempts to mimic the behavior of the target hardware.

## Architecture

### Primitive Simulation (`prims/`)
Each hardware primitive (e.g., ALU, Memory Unit) has a corresponding software model in `prims/`.

### Control Loop
How the emulator iterates through cycles or steps.

### Memory Model
How data storage is simulated.
