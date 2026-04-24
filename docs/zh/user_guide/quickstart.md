# 快速开始

本节将指导您使用 T-Stack 工具链运行一个简单的示例。

## 前置条件
*   已安装 Python 3.x。
*   已安装 `requirements.txt` 中列出的依赖项。
TODO: 确定python版本，整理requirements

## 基本使用流程

您可以在 `tests/emulator/prims/add_test.py` , `tests/analyser/prims/add_test.py` 和 `tests/program/networks/resnet9/resnet9_test.py` 中找到基本示例。

### 1. 前端：定义图 (Define the Graph)
以简单的add算子为例
```python
# 建立图
graph = Graph()

# 生成输入数据
data_1 = torch.randn(2, 3, 32).to(torch.bfloat16)
data_2 = torch.randn(2, 3, 32).to(torch.bfloat16)

# 生成数据节点
input_data_1 = Data(name="input1", dtype=DataType.BF16, shape=(2, 3, 32), payload=data_1)
input_data_2 = Data(name="input2", dtype=DataType.BF16, shape=(2, 3, 32), payload=data_2)
output_data = Data(name="output")

add_op = Add("add_op", {"output_dtype": DataType.BF16, "bc_mode": 0, "scalar": 0, "add_or_sub": 0})

# 在图中添加节点
graph.add_node(input_data_1)
graph.add_node(input_data_2)
graph.add_node(output_data)
graph.add_node(add_op)

# 连接节点和数据
graph.connect(input_data_1.name, add_op.name)
graph.connect(input_data_2.name, add_op.name)
graph.connect(add_op.name, output_data.name)

# 添加、连接停止原语
stop_op = Stop("stop_op", attrs={"jump_addr": 0, "relative": 0, "jump": 1})
graph.add_node(stop_op)
graph.connect_control(add_op.name, stop_op.name)

# 形状推理
graph.infer()
```

### 2. 中端：图划分、核映射与通信生成
```python
hardware_graph = HardwareGraph(graph)
hardware_graph.gen_memref_for_all_data()
hardware_graph.set_core_id_for_nodes(hardware_graph.all_nodes(), core_id=(0, 0))  # 手动设置每个节点（算子或数据）的核ID
hardware_graph.gen_communication_ops()
```

### 3. 后端：存储分配、原语顺序生成
```python
allocator = MemoryAllocator(hardware_graph)
hardware_graph.gen_para_nodes()
allocator.allocate_memory(mem_per_core=16384, reserved_space=8, non_overwritable_patterns=[], incremental=False)
op_sch = OperationScheduler(hardware_graph)
op_lists = op_sch.build_core_op_lists(try_parallel=True)
deps = op_sch.build_deps_for_ops(8)
```

### 4. 代码生成/仿真分析
#### 4-1（面向实际芯片/Program）代码/配置生成
请参考 `tests/program/networks/resnet9/resnet9_test.py`
注意：`ConfigGenerator`生成的配置的格式与验证/使用场景和系统设计相关，在实际芯片系统设计/测试完成之前，可能会有较大变化。目前有`generate_all_configs`和`generate_lvds_c_configs`等多个不同接口
```python
code_gen = CodeGenerator(hardware_graph, op_lists, deps)
codes = code_gen.generate_code()

config_gen = ConfigGenerator(hardware_graph, codes, output_dir="testcases/resnet9_single_core/")
config_gen.generate_all_configs()
# config_gen.generate_lvds_c_configs()
print("ResNet9 test graph 完成。")
```

#### 4-2 功能仿真
请参考 `tests/emulator/prims/add_test.py`

```python
core_array = EmulatorArray(config=load_config_from_yaml("core/simulator/configs/basic_config.yaml"), array_size=(1, 1))
core_array.deploy(hardware_graph, op_lists, deps)

core_array.run()

outputs = core_array.get_outputs(get_output_list(hardware_graph, ["output"]))
```

#### 4-2 性能分析
请参考 `tests/program/networks/resnet9/resnet9_test.py`

```python
chip_array = ChipArray((1, 1), (1, 1), config=load_config_from_yaml("core/simulator/configs/basic_config.yaml"))
    
chip_array.deploy(hardware_graph, op_lists, deps)

engine = Engine(chip_array, load_config_from_yaml("core/simulator/configs/basic_config.yaml"))
engine.run()
engine.printUtilizations()
```

## 使用思路
工具链主要为芯片服务，同时可以用于算法设计和架构探索。

通过Compiler完成的编程和编译后，可以使用emulator先验证模型的正确性；然后使用analyser评估性能；最后使用ConfigGenerator生成配置文件，部署到芯片上。