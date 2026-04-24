# Compiler Design

The compiler (Program module) is responsible for translating the user's computational graph into a hardware-executable configuration.

## Key Components

### Graph Representation (`core_components/graph.py`)
Describes how the node-link graph is stored and manipulated.

### Operation Definitions (`operations/`)
How new operations are added. Each operation likely needs a definition of its inputs, outputs, and hardware mapping logic.

### Scheduler (`core_components/operation_scheduler.py`)
Details on how operations are scheduled for execution (e.g., topological sort, dependency analysis).

### Config Generation (`core_components/config_gen.py`)
The process of serializing the scheduled graph into the final configuration format.
