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

def construct_router_graph() -> Graph:
    graph = Graph()
    
    # 32 elements of BF16 (1 packet usually holds 32 bytes or similar? 
    # PrimSendRecv sends units. 
    # SendMsg default pack_per_rhead=inputs[0].memref.length-1
    # inputs[0].memref.length is number of elements if addressing is by element.
    # Data memref length is size in bytes / 8? No. 
    # Let's check Data.memref.
    pass

    data_shape = (1, 16, 16) # 32 elements
    data_payload = torch.randn(data_shape).to(torch.bfloat16)
    
    # Source Data on Core (0,0)
    input_data = Data(name="input_data", dtype=DataType.BF16, shape=data_shape, payload=data_payload)
    
    # SendOp on Core (0,0) sending to (0,1)
    send_op = SendOp("send_op", attrs={"source": (0, 0), "dest": (0, 1), "tag": 1})
    
    # 手动配置 Send 原语和 Msg
    # 配置 Send 原语参数
    send_prim_config = {
        'cell_or_neuron': 0,      # 0表示cell模式
        'pack_head_num': 0,       # pack head数量
    }
    
    # 配置 Msg 列表（可以配置多个 Msg）
    msg_configs = [
        {
            'S': 0,                 # S 标志
            'T': 0,                 # T 标志
            'E': 0,                 # E 标志
            'Q': 0,                 # Q 标志
            'LVDS': 0,              # LVDS 标志
            'Y': 0,                 # Y 坐标偏移 (dest[0] - source[0])
            'X': 1,                 # X 坐标偏移 (dest[1] - source[1])
            # 'A0': 4,                # A0 字段
            'A0': 0,                # A0 字段
            'pack_per_rhead': 16 - 1,    # 每个 rhead 的 pack 数量
            # 'A_offset': -7,          # A 偏移
            'A_offset': 1,          # A 偏移
            'Const': 0,             # 常量字段
            'handshake': 1,         # 握手标志
            'tag_id': 1,            # tag ID
            'en': 1,                # 使能标志
            'sparse': 0             # 稀疏标志
        }
        # 如果需要发送到多个目的地，可以添加更多 Msg 配置
        # {
        #     'S': 1, 'T': 0, 'E': 0, 'Q': 0, 'LVDS': 0,
        #     'Y': 1, 'X': 0, 'A0': 0, 'pack_per_rhead': 0,
        #     'A_offset': 0, 'Const': 0, 'handshake': 1,
        #     'tag_id': 1, 'en': 1, 'sparse': 0
        # }
    ]
    
    # 调用手动配置方法
    send_op.configure_send_manual(send_prim_config, msg_configs)
    
    # Dest Data on Core (0,1)
    output_data = Data(name="output_data", dtype=DataType.BF16, shape=data_shape)
    
    # RecvOp on Core (0,1) receiving from (0,0)
    recv_op = RecvOp("recv_op", attrs={"source": (0, 0), "tag": 1})
    
    # 手动配置 Recv 原语
    recv_prim_config = {
        'CXY': 0,               # CXY 标志
        'mc_y': 0,              # multicast Y 坐标
        'mc_x': 0,              # multicast X 坐标
        'tag_id': 1,            # tag ID（需要与 SendOp 的 tag_id 匹配）
        'end_num': 0            # 结束数量
    }
    recv_op.configure_recv_manual(recv_prim_config)
    
    
    # Add nodes
    graph.add_node(input_data)
    graph.add_node(send_op)
    graph.add_node(recv_op)
    graph.add_node(output_data)
    
    # Connect
    graph.connect(input_data.name, send_op.name)
    graph.connect(send_op.name, recv_op.name)
    graph.connect(recv_op.name, output_data.name)
    
    # Stops
    stop_op_0 = Stop("stop_op_0", attrs={"jump_addr": 0, "relative": 0, "jump": 1})
    stop_op_1 = Stop("stop_op_1", attrs={"jump_addr": 0, "relative": 0, "jump": 1})
    
    graph.add_node(stop_op_0)
    graph.add_node(stop_op_1)
    
    # Control
    graph.connect_control(send_op.name, stop_op_0.name)
    graph.connect_control(recv_op.name, stop_op_1.name)
    
    graph.infer()
    
    return graph, data_payload

if __name__ == "__main__":
    graph, true_output = construct_router_graph()
    
    hardware_graph = HardwareGraph(graph)
    hardware_graph.gen_memref_for_all_data()

    # Manually set Core IDs
    # Core (0,0)
    hardware_graph.set_core_id("input_data", (0, 0))
    hardware_graph.set_core_id("send_op", (0, 0))
    hardware_graph.set_core_id("stop_op_0", (0, 0))
    
    # Core (0,1)
    hardware_graph.set_core_id("recv_op", (0, 1))
    hardware_graph.set_core_id("output_data", (0, 1))
    hardware_graph.set_core_id("stop_op_1", (0, 1))

    allocator = MemoryAllocator(hardware_graph)
    hardware_graph.gen_para_nodes() # Generates params
    
    # Params also need to have valid Core IDs?
    # SendOp.para_node creates "send_op.params".
    # HardwareGraph.gen_para_nodes adds it. But default ID is (0,0).
    # Since send_op is on (0,0), it should be fine.
    
    allocator.allocate_memory(mem_per_core=16384, reserved_space=8, non_overwritable_patterns=[], incremental=False)
    
    op_sch = OperationScheduler(hardware_graph)
    op_lists = op_sch.build_core_op_lists(try_parallel=True)
    deps = op_sch.build_deps_for_ops(8)
    
    print("\nStarting Simulation...")
    # array_size=(1, 2) to accommodate (0,1)
    core_array = EmulatorArray(config=load_config_from_yaml("core/simulator/configs/basic_config.yaml"), array_size=(1, 2))
    
    core_array.deploy(hardware_graph, op_lists, deps)
    
    core_array.run()
    
    outputs = core_array.get_outputs(get_output_list(hardware_graph, ["output_data"]))
    
    # print("\nTrue Output:\n", true_output)
    # print("Simulated Output:\n", outputs["output_data"])
    
    diff = torch.mean(torch.abs(outputs["output_data"] - true_output))
    print(f"Difference: {diff}")
    
    if diff < 1e-4:
        print("Router primitive test passed!")
    else:
        print("Router primitive test failed!")
