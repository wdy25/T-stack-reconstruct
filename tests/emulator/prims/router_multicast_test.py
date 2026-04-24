import torch
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from core.ir.data import Data, DataType
from core.ir.graph import Graph
from core.ir.hardware_graph import HardwareGraph
from core.compiler.memory_allocator import MemoryAllocator
from core.compiler.operation_scheduler import OperationScheduler
from core.ir.communication_op import SendOp, RecvOp
from core.ir.operations.stop import Stop
from core.simulator.emulator.chip import Chip as EmulatorArray
from core.simulator.emulator.output_if import get_output_list
from basics.utils import load_config_from_yaml

def construct_router_multicast_graph() -> Graph:
    graph = Graph()
    
    # Same data shape as router_test.py
    data_shape = (1, 16, 16) 
    data_payload = torch.randn(data_shape).to(torch.bfloat16)
    
    # ---------------------------------------------------------
    # Core (0, 2): Source
    # ---------------------------------------------------------
    input_data = Data(name="input_data", dtype=DataType.BF16, shape=data_shape, payload=data_payload)
    
    # SendOp on Core (0,2) sending to (0,1)
    send_op = SendOp("send_op", attrs={"source": (0, 2), "dest": (0, 1), "tag": 1})
    
    send_prim_config = {
        'cell_or_neuron': 0,      # mode
        'pack_head_num': 0,       # pack head num
    }
    
    # Msg config for sending to (0,1) from (0,2)
    # Relative offset: Y=0, X = 1-2 = -1
    msg_configs = [
        {
            'S': 0, 
            'T': 0, 
            'E': 0, 
            'Q': 1, 
            'LVDS': 0,
            'Y': 0,                 # dest_y(0) - src_y(0)
            'X': -1,                # dest_x(1) - src_x(2)
            'A0': 0,                
            'pack_per_rhead': 16 - 1,    
            'A_offset': 1,          
            'Const': 0, 
            'handshake': 1,         
            'tag_id': 1,            
            'en': 1,                
            'sparse': 0             
        }
    ]
    
    send_op.configure_send_manual(send_prim_config, msg_configs)
    
    # ---------------------------------------------------------
    # Core (0, 1): Intermediate (Recv + Multicast)
    # ---------------------------------------------------------
    output_data_1 = Data(name="output_data_1", dtype=DataType.BF16, shape=data_shape)
    
    # RecvOp on (0,1) receiving from (0,2)
    # configured with multicast to (0,0)
    recv_op_1 = RecvOp("recv_op_1", attrs={"source": (0, 2), "tag": 1})
    
    recv_prim_config_1 = {
        'CXY': 1,               # Enable Multicast
        'mc_y': 0,              # Offset to multicast target (0,0) from (0,1): 0-0=0
        'mc_x': -1,             # Offset to multicast target (0,0) from (0,1): 0-1=-1
        'tag_id': 1,            
        'end_num': 0            
    }
    recv_op_1.configure_recv_manual(recv_prim_config_1)
    
    # ---------------------------------------------------------
    # Core (0, 0): Destination
    # ---------------------------------------------------------
    output_data_0 = Data(name="output_data_0", dtype=DataType.BF16, shape=data_shape)
    
    # RecvOp on (0,0) receiving from (0,1) (via multicast)
    recv_op_0 = RecvOp("recv_op_0", attrs={"source": (0, 1), "tag": 1})
    
    recv_prim_config_0 = {
        'CXY': 0,               # No multicast
        'mc_y': 0,              
        'mc_x': 0,              
        'tag_id': 1,            
        'end_num': 0            
    }
    recv_op_0.configure_recv_manual(recv_prim_config_0)
    
    
    # ---------------------------------------------------------
    # Add nodes to graph
    # ---------------------------------------------------------
    graph.add_node(input_data)
    graph.add_node(send_op)
    
    graph.add_node(recv_op_1)
    graph.add_node(output_data_1)
    
    graph.add_node(recv_op_0)
    graph.add_node(output_data_0)
    
    # ---------------------------------------------------------
    # Connect Data Edges
    # ---------------------------------------------------------
    graph.connect(input_data.name, send_op.name)
    graph.connect(send_op.name, recv_op_1.name)
    
    # Connect recv output to local data
    graph.connect(recv_op_1.name, output_data_1.name)
    
    # Connect recv_op_1 to recv_op_0 to represent dependency/flow?
    # In router_test, send connects to recv. 
    # Here flow is send -> recv1 -> recv0 (multicast)
    # We connect recv_op_1 to recv_op_0 to ensure scheduling if needed, or logical flow.
    # However, standard practice might differ. We will assume logical connection follows path.
    # But recv_op_1 produces data, recv_op_0 does not take data input in IR (it takes network input).
    # We'll rely on placement and implicit control, but if graph.connect checks ports, we might need care.
    # Given send_op -> recv_op works (send consumes data), and recv producer data.
    # I'll not connect recv_op_1 to recv_op_0 with a data edge to avoid port mismatch if any.
    # Instead, I can connect them via control if needed, but separate components are usually fine.
    # Actually, let's just leave them as islands connected by deployment/tag matching, 
    # or follow router_test pattern where send->recv is connected.
    # I'll connect send_op -> recv_op_0 as well? No, that implies direct send.
    # I will stick to minimal connections.
    
    graph.connect(recv_op_0.name, output_data_0.name)
    
    # ---------------------------------------------------------
    # Stops and Control
    # ---------------------------------------------------------
    stop_op_2 = Stop("stop_op_2", attrs={"jump_addr": 0, "relative": 0, "jump": 1})
    stop_op_1 = Stop("stop_op_1", attrs={"jump_addr": 0, "relative": 0, "jump": 1})
    stop_op_0 = Stop("stop_op_0", attrs={"jump_addr": 0, "relative": 0, "jump": 1})
    
    graph.add_node(stop_op_2)
    graph.add_node(stop_op_1)
    graph.add_node(stop_op_0)
    
    # Control deps to ensure Ops run before Stop
    graph.connect_control(send_op.name, stop_op_2.name)
    graph.connect_control(recv_op_1.name, stop_op_1.name)
    graph.connect_control(recv_op_0.name, stop_op_0.name)
    
    graph.infer()
    
    return graph, data_payload

if __name__ == "__main__":
    graph, true_output = construct_router_multicast_graph()
    
    hardware_graph = HardwareGraph(graph)
    hardware_graph.gen_memref_for_all_data()

    # Manually set Core IDs
    # Core (0,2)
    hardware_graph.set_core_id("input_data", (0, 2))
    hardware_graph.set_core_id("send_op", (0, 2))
    hardware_graph.set_core_id("stop_op_2", (0, 2))
    
    # Core (0,1)
    hardware_graph.set_core_id("recv_op_1", (0, 1))
    hardware_graph.set_core_id("output_data_1", (0, 1))
    hardware_graph.set_core_id("stop_op_1", (0, 1))

    # Core (0,0)
    hardware_graph.set_core_id("recv_op_0", (0, 0))
    hardware_graph.set_core_id("output_data_0", (0, 0))
    hardware_graph.set_core_id("stop_op_0", (0, 0))

    allocator = MemoryAllocator(hardware_graph)
    hardware_graph.gen_para_nodes() 
    
    allocator.allocate_memory(mem_per_core=16384, reserved_space=8, non_overwritable_patterns=[], incremental=False)
    
    op_sch = OperationScheduler(hardware_graph)
    op_lists = op_sch.build_core_op_lists(try_parallel=True)
    deps = op_sch.build_deps_for_ops(8)
    
    print("\nStarting Multicast Simulation...")
    # array_size=(1, 3) to accommodate (0,0), (0,1), (0,2)
    core_array = EmulatorArray(config=load_config_from_yaml("core/simulator/configs/basic_config.yaml"), array_size=(1, 3))
    
    core_array.deploy(hardware_graph, op_lists, deps)
    
    core_array.run()
    
    outputs = core_array.get_outputs(get_output_list(hardware_graph, ["output_data_1", "output_data_0"]))
    
    diff_1 = torch.mean(torch.abs(outputs["output_data_1"] - true_output))
    diff_0 = torch.mean(torch.abs(outputs["output_data_0"] - true_output))
    
    print(f"Diff Core(0,1): {diff_1}")
    print(f"Diff Core(0,0): {diff_0}")
    
    if diff_1 < 1e-4 and diff_0 < 1e-4:
        print("Router multicast test passed!")
    else:
        print("Router multicast test failed!")
