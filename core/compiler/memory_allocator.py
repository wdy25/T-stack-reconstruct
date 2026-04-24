"""
内存地址分配模块

该模块为hardware_graph中的data节点分配内存地址，考虑数据依赖关系、
核内存限制、预留地址空间以及inplace操作等因素。
"""

from typing import Dict, List, Set, Optional, Tuple, Any
from core.ir.hardware_graph import HardwareGraph
from core.ir.data import Data, MemBlock, ConcatData
from core.ir.operation import Operation
import hashlib
import warnings
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class MemoryAllocationError(Exception):
    """内存分配相关的异常"""
    pass

class VirtualMemory:
    """虚拟内存空间，表示一整块内存区域的分配状态"""
    def __init__(self, total_size: int):
        self.total_size = total_size
        self.free_blocks: List[Tuple[int, int]] = [(0, total_size - 1)]  # (start_addr, end_addr)
    
    def allocate_wo_check(self, addr, size):
        """无检查地分配指定地址和大小的内存块"""
        new_free_blocks = []
        for start_addr, end_addr in self.free_blocks:
            if end_addr < addr or start_addr > addr + size - 1:
                # 不重叠的块保持不变
                new_free_blocks.append((start_addr, end_addr))
            else:
                # 重叠的块需要分割
                if start_addr < addr:
                    # 前半部分还是空闲的
                    new_free_blocks.append((start_addr, addr - 1))
                if end_addr > addr + size - 1:
                    # 后半部分还是空闲的
                    new_free_blocks.append((addr + size, end_addr))
        self.free_blocks = new_free_blocks
        self._merge_free_blocks()
    
    def _merge_free_blocks(self):
        self.free_blocks.sort(key=lambda x: x[0])
        
        merged_blocks = []
        
        i = 0
        while i < len(self.free_blocks):
            current_start, current_end = self.free_blocks[i]
            j = i + 1
            while j < len(self.free_blocks):
                next_start, next_end = self.free_blocks[j]
                if next_start <= current_end + 1:
                    # 可以合并
                    current_end = max(current_end, next_end)
                    j += 1
                else:
                    break
            merged_blocks.append((current_start, current_end))
            i = j     

        self.free_blocks = merged_blocks
    
    def try_allocate(self, size):
        """尝试在指定地址分配指定大小的内存块，成功返回True，失败返回False"""
        for start_addr, end_addr in self.free_blocks:
            block_size = end_addr - start_addr + 1
            if block_size >= size:
                # 找到合适的块，进行分配
                allocated_addr = start_addr
                self.allocate_wo_check(allocated_addr, size)
                return allocated_addr
        return None
    
    def try_allocate_at(self, addr, size) -> bool:
        """尝试在指定地址分配指定大小的内存块，成功返回True，失败返回False"""
        for start_addr, end_addr in self.free_blocks:
            if start_addr <= addr and end_addr >= addr + size - 1:
                # 找到合适的块，进行分配
                self.allocate_wo_check(addr, size)
                return True
        return False
    

class MemoryAllocator:
    """内存分配器类
    
    负责为hardware_graph中的data节点分配内存地址，支持：
    1. 数据依赖关系管理
    2. 每核内存限制
    3. 预留地址空间
    4. inplace操作优化
    """
    
    def __init__(self, hardware_graph: HardwareGraph):
        self.hardware_graph = hardware_graph
        self.core_allocations: Dict[Tuple[int, int], Dict[str, Tuple[int, int]]] = {}
        self.lifecycle_records: Dict[Tuple[int, int], Dict[str, Any]] = {}  # 节点生命周期记录
        self.nodes_without_memref: List[str] = []  # 记录没有memref的节点
        self.non_overwritable_patterns: List[str] = []  # 不可覆盖的数据模式列表
        self.core_memory_info: Dict[Tuple[int, int], Dict[str, Any]] = {}  # 每核内存信息

    def allocate_memory(
        self, 
        mem_per_core: int, 
        reserved_space: int = 0,
        incremental: bool = False,
        non_overwritable_patterns: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """为所有data节点分配内存地址
        
        Args:
            mem_per_core: 每个核上的最大地址空间
            reserved_space: 每个核从地址0开始预留的地址空间
            incremental: 是否增量分配（读取已有地址，只为未分配的节点分配新地址）
            non_overwritable_patterns: 不可覆盖的数据节点名称模式列表
            
        Returns:
            Dict[str, int]: 节点名称到地址的映射
            
        Raises:
            MemoryAllocationError: 当内存分配失败时
        """
        # 重置生命周期记录
        self.lifecycle_records = {}
        
        # 设置不可覆盖模式
        self.non_overwritable_patterns = non_overwritable_patterns or []
        
        # 记录所有没有memref的节点（不可分配）
        self.nodes_without_memref = []
        for nid in self.hardware_graph._nodes:
            if isinstance(self.hardware_graph._nodes[nid], Data) and self.hardware_graph._nodes[nid].memref is None:
                if nid not in self.nodes_without_memref:
                    self.nodes_without_memref.append(nid)
        
        # 初始化每个核的分配信息
        self._initialize_core_allocations(mem_per_core, reserved_space)
        
        # 如果是增量分配，先读取已有地址
        if incremental:
            # Dict[str, int]
            existing_addresses = self._read_existing_addresses()
        else:
            existing_addresses = {}
        
        # 获取所有核ID
        all_cores = self.hardware_graph.get_all_core_ids()
        
        # 为每个核分别分配内存
        address_mapping = {}
        for core_id in all_cores:
            core_addresses = self._allocate_memory_for_core(core_id, existing_addresses, incremental)
            address_mapping.update(core_addresses)
        
        # 更新hardware_graph中的memref地址
        self._update_memref_addresses(address_mapping)
        
        # 输出没有memref的节点提示
        if self.nodes_without_memref:
            print(f"\n警告：以下 {len(self.nodes_without_memref)} 个节点没有memref，未分配地址：")
            for node_id in self.nodes_without_memref:
                print(f"  - {node_id}")
            print("建议先调用 hardware_graph.gen_memref_for_all_data() 生成memref")
        
        return address_mapping
    
    def _record_lifecycle_event(self, node_id: str, event_type: str, addr: int = None, size: int = 0, time_step: int = None):
        """记录数据生命周期事件（简化版）
        
        Args:
            node_id: 节点ID
            event_type: 事件类型 ('allocate', 'free')
        """
        core_id = self.hardware_graph.get_core_id(node_id)
        
        if core_id not in self.lifecycle_records:
            self.lifecycle_records[core_id] = {}
        
        if node_id not in self.lifecycle_records[core_id]:
            self.lifecycle_records[core_id][node_id] = {
                'allocate_time': None,
                'free_time': None,
                'core_id': None,
                'address': None,
                'size': 0
            }
        
        
        if event_type == 'allocate':
            self.lifecycle_records[core_id][node_id]['allocate_time'] = time_step
            # 获取节点信息
            node = self.hardware_graph.node(node_id)
            if isinstance(node, Data):
                self.lifecycle_records[core_id][node_id]['core_id'] = core_id
                assert addr is not None, "地址不能为空"
                self.lifecycle_records[core_id][node_id]['address'] = addr
                assert size > 0, "大小必须大于0"
                self.lifecycle_records[core_id][node_id]['size'] = size
            elif isinstance(node, ConcatData):
                self.lifecycle_records[core_id][node_id]['core_id'] = core_id
                assert addr is not None, "地址不能为空"
                self.lifecycle_records[core_id][node_id]['address'] = addr
                assert size > 0, "大小必须大于0"
                self.lifecycle_records[core_id][node_id]['size'] = size
        elif event_type == 'free':
            self.lifecycle_records[core_id][node_id]['free_time'] = time_step
            
    def visualize_lifecycle(self, save_path: str = None, show_details: bool = True, cores_to_visualize: List[Tuple[int, int]] = None):
        """可视化数据生命周期（内存地址时间线视图）
        
        Args:
            save_path: 保存路径
            show_details: 是否显示详细信息
        """
        if not self.lifecycle_records:
            print("没有生命周期记录数据，请先运行内存分配")
            return
        
        # 按核心分组
        if cores_to_visualize:
            cores = set(cores_to_visualize)
        else:
            cores = set([core_id for core_id in self.lifecycle_records])
        num_cores = len(cores) if cores else 1
        
        # 创建图表 - 每个核心一个子图，设置足够的高度来显示内存地址
        fig_height = max(8, num_cores * 10)  # 每个核心至少10英寸高度
        fig, axes = plt.subplots(num_cores, 1, figsize=(16, fig_height))
        if num_cores == 1:
            axes = [axes]
        
        sorted_cores = sorted(cores) if cores else [None]
        
        for i, core_id in enumerate(sorted_cores):
            ax = axes[i]
            
            # 筛选该核心的数据
            core_records = {node_id: record for node_id, record in self.lifecycle_records[core_id].items() 
                           if record['allocate_time'] is not None}
            
            max_time_step = max(record['allocate_time'] for record in core_records.values()) + 1
            
            if not core_records:
                ax.text(0.5, 0.5, f'Core {core_id}: No Data Records', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # 设置颜色
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', 
                     '#FFB6C1', '#98FB98', '#87CEEB', '#DEB887', '#F0E68C', '#E6E6FA']
            
            # 找到地址范围
            min_addr = min(record['address'] for record in core_records.values() if record['address'] is not None)
            max_addr = max(record['address'] + record['size'] for record in core_records.values() 
                          if record['address'] is not None and record['size'] > 0)
            
            # 为每个数据节点绘制矩形
            for node_id, record in core_records.items():
                if record['address'] is None or record['size'] <= 0:
                    continue
                    
                # 计算时间跨度
                start_time = record['allocate_time']
                end_time = record['free_time'] if record['free_time'] is not None else max_time_step
                
                # 绘制内存块矩形
                rect = patches.Rectangle(
                    (start_time, record['address']), 
                    end_time - start_time, 
                    record['size'],
                    linewidth=1, 
                    edgecolor='black',
                    facecolor=colors[int.from_bytes(hashlib.sha256(node_id.encode('utf-8')).digest(), 'big') % len(colors)], 
                    alpha=0.7
                )
                ax.add_patch(rect)
                
                # 添加标签
                if show_details:
                    # 计算标签位置 - 统一放在矩形最左侧
                    label_x = start_time + 0.05  # 稍微偏移避免贴边
                    label_y = record['address'] + record['size'] / 2
                    
                    # 构建标签文本
                    label_lines = [node_id]
                    label_lines.append(f"Addr: {record['address']}")
                    label_lines.append(f"Size: {record['size']}")
                    
                    # 根据矩形大小调整字体 - 确保标签不超出矩形范围
                    rect_width = end_time - start_time
                    rect_height = record['size']
                    
                    # 估算文本尺寸并计算合适的字号
                    def estimate_text_size(text, fontsize):
                        # 简单估算：每个字符宽度约为fontsize*0.6，高度约为fontsize*1.2
                        lines = text.split('\n')
                        max_line_width = max(len(line) for line in lines) * fontsize * 0.5
                        text_height = len(lines) * fontsize * 1
                        return max_line_width, text_height
                    
                    # 根据矩形面积和最小尺寸确定显示内容和字号
                    min_dimension = min(rect_width, rect_height)
                    
                    if rect_width > 1 and rect_height > 50 and min_dimension > 0.8:  # 大矩形显示完整信息
                        label_text = '\n'.join(label_lines)
                        # 计算能容纳完整文本的最大字号
                        max_fontsize = 8
                        for fs in range(max_fontsize, 3, -1):
                            text_w, text_h = estimate_text_size(label_text, fs)
                            if text_w <= rect_width * 0.9 and text_h <= rect_height * 0.9:
                                fontsize = fs
                                break
                        else:
                            fontsize = 4
                    elif rect_width > 0.5 and rect_height > 40 and min_dimension > 0.3:  # 中矩形显示简化信息
                        label_text = f"{node_id}\n{record['size']}"
                        # 计算能容纳简化文本的最大字号
                        max_fontsize = 7
                        for fs in range(max_fontsize, 3, -1):
                            text_w, text_h = estimate_text_size(label_text, fs)
                            if text_w <= rect_width * 0.9 and text_h <= rect_height * 0.9:
                                fontsize = fs
                                break
                        else:
                            fontsize = 4
                    elif rect_width > 0.3 and rect_height > 10 and min_dimension > 0.2:  # 小矩形只显示名称
                        label_text = node_id
                        # 计算能容纳节点名的最大字号
                        max_fontsize = 6
                        for fs in range(max_fontsize, 3, -1):
                            text_w, text_h = estimate_text_size(label_text, fs)
                            if text_w <= rect_width * 0.9 and text_h <= rect_height * 0.9:
                                fontsize = fs
                                break
                        else:
                            fontsize = 4
                    else:  # 太小的矩形不显示标签
                        label_text = ""
                        fontsize = 4
                    
                    if label_text:
                        ax.text(label_x, label_y, label_text, 
                               ha='left', va='center', 
                               fontsize=fontsize, 
                               rotation=0,
                               color='black')
                
            
            # 设置坐标轴
            ax.set_xlim(-0.5, max_time_step + 0.5)
            ax.set_ylim(min_addr - 50, max_addr + 50)
            ax.set_xlabel('Time Steps (Allocation Events)')
            ax.set_ylabel('Memory Address')
            ax.set_title(f'Memory Address Timeline - Core {core_id}')
            ax.grid(True, alpha=0.3)
            
            # 添加时间步标记
            for j in range(0, max_time_step + 1):
                ax.axvline(x=j, color='gray', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"生命周期可视化已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _initialize_core_allocations(self, mem_per_core: int, reserved_space: int):
        """初始化每个核的内存分配信息"""
        all_cores = self.hardware_graph.get_all_core_ids()
        assert reserved_space < mem_per_core, "预留空间必须小于每核总内存"
        for core_id in all_cores:
            self.core_allocations[core_id] = {}
            if reserved_space > 0:
                self.core_allocations[core_id]['reserved'] = (0, reserved_space - 1)
            self.lifecycle_records[core_id] = {}
            if reserved_space > 0:
                self.lifecycle_records[core_id]['reserved'] = {
                    'allocate_time': 0,
                    'free_time': None,
                    'core_id': core_id,
                    'address': 0,
                    'size': reserved_space
                }
            self.core_memory_info[core_id] = {
                'total_memory': mem_per_core,
                'reserved_space': reserved_space,
            }
    
    def _read_existing_addresses(self) -> Dict[str, int]:
        """读取已有的地址分配
        
        Returns:
            Dict[str, int]: 已有的节点名称到地址的映射
        """
        existing_addresses = {}
        
        for node_id, node in self.hardware_graph._nodes.items():
            if isinstance(node, Data) and node.memref and node.memref.addr != -1 and node.memref.addr is not None:
                existing_addresses[node_id] = node.memref.addr
            if isinstance(node, ConcatData) and node.inferred_memref and node.inferred_memref.addr != -1 and node.inferred_memref.addr is not None:
                existing_addresses[node_id] = node.inferred_memref.addr
        
        return existing_addresses
    
    def _allocate_memory_for_core(
        self, 
        core_id: Tuple[int, int], 
        existing_addresses: Dict[str, int] = None,
        incremental: bool = False
    ) -> Dict[str, int]:
        """为指定核分配内存
        
        Args:
            core_id: 核ID
            existing_addresses: 已有的地址分配
            incremental: 是否增量分配
            
        Returns:
            Dict[str, int]: 该核上节点名称到地址的映射
        """
        if existing_addresses is None or not incremental:
            existing_addresses = {}
        
        # 获取该核上的所有节点
        core_nodes = self.hardware_graph.get_nodes_by_core(core_id)
        
        # 分离数据节点、拼接节点和操作节点
        data_nodes: List[str] = []
        concat_nodes: List[str] = []
        operation_nodes: List[str] = []
        
        for node_id in core_nodes:
            node_type = self.hardware_graph.kind_of(node_id)
            if node_type == "data":
                data_nodes.append(node_id)
            elif node_type == "concat":
                concat_nodes.append(node_id)
            elif node_type == "operation":
                operation_nodes.append(node_id)
        
        # 识别 concat 分组：将属于 concat 输入的数据节点从直接分配队列中移除，改为整体由 concat 节点分配
        grouped_data_to_concat: Dict[str, str] = {}
        concat_inputs_by_port: Dict[str, Dict[int, str]] = {}
        for cid in concat_nodes:
            # 获取该 concat 的输入端口映射（端口->前驱数据节点ID）
            preds = self.hardware_graph.predecessors(cid)
            ports = self.hardware_graph.in_ports(cid)
            port_map: Dict[int, str] = {}
            for pred, port in zip(preds, ports):
                # 仅接受数据节点作为 concat 输入
                if self.hardware_graph.kind_of(pred) == "data":
                    p = 0 if port is None else port
                    port_map[p] = pred
                    grouped_data_to_concat[pred] = cid
            # 端口表按端口顺序完整性不在此强校验
            concat_inputs_by_port[cid] = port_map

        # 将被 concat 覆盖的 data 节点剔除，仅保留非分组 data，以及 concat 节点本身
        data_nodes = [nid for nid in data_nodes if nid not in grouped_data_to_concat]

        # 如果是增量分配，记录已有地址但不预先占用内存
        core_addresses = {}
        nodes_to_allocate = []
        
        # 参与分配的节点包括：非分组 data 节点 + concat 节点
        alloc_candidates = data_nodes + concat_nodes

        for node_id in alloc_candidates:
            if incremental and node_id in existing_addresses:
                # 记录已有地址，但不预先占用内存
                core_addresses[node_id] = existing_addresses[node_id]
            else:
                # 需要新分配地址的节点
                nodes_to_allocate.append(node_id)
        
        # 如果没有需要分配的节点，直接返回
        if not nodes_to_allocate:
            return core_addresses
        
        # 构建数据依赖图（包括所有需要参与分配的节点；被 concat 覆盖的原始 data 不加入）
        dependency_graph = self._build_dependency_graph(data_nodes + concat_nodes, operation_nodes, grouped_data_to_concat)
        
        # 获取拓扑排序(只包含有memref的节点)
        allocation_order = self._topological_sort(dependency_graph)
        
        # 按顺序处理内存分配
        for node_id in allocation_order:
            if node_id in alloc_candidates:
                # 跳过没有memref的节点
                node_size = self._get_node_memory_size(node_id)
                if node_size > 0:
                    address = self._allocate_node_memory(node_id, core_id, dependency_graph, existing_addresses)
                    core_addresses[node_id] = address

        return core_addresses
    
    def _build_dependency_graph(
        self, 
        data_nodes: List[str], 
        operation_nodes: List[str],
        grouped_data_to_concat: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """构建数据依赖图
        
        Args:
            data_nodes: 数据节点列表
            operation_nodes: 操作节点列表
            
        Returns:
            Dict[str, Dict[str, Any]]: 依赖图，包含生命周期和inplace信息
        """
        dependency_graph = {}
        
        # 初始化所有数据节点
        for node_id in data_nodes:
            node_size = self._get_node_memory_size(node_id)
            dependency_graph[node_id] = {
                'predecessors': set(),  # 依赖的节点
                'successors': set(),    # 被依赖的节点
                'last_use': None,       # 最后使用该数据的操作
                'can_be_overwritten_by': set(),  # 可以被哪些节点覆盖
                'size': node_size,
                'has_memref': node_size > 0  # 是否有有效的memref
            }
        
        # 分析数据流依赖
        for op_node_id in operation_nodes:
            operation = self.hardware_graph.node(op_node_id)
            is_inplace = self._is_inplace_operation(operation)
            
            # 获取输入和输出数据节点
            input_data_nodes: List[str] = []
            output_data_nodes = []
            
            for pred_id in self.hardware_graph.predecessors(op_node_id):
                if self.hardware_graph.kind_of(pred_id) == "data":
                    # 若该 data 属于某 concat，则以 concat 节点替代
                    if grouped_data_to_concat and pred_id in grouped_data_to_concat:
                        cid = grouped_data_to_concat[pred_id]
                        if cid in data_nodes and cid not in input_data_nodes:
                            input_data_nodes.append(cid)
                    elif pred_id in data_nodes:
                        input_data_nodes.append(pred_id)
                if self.hardware_graph.kind_of(pred_id) == "view":
                    # 处理view节点的输入
                    for view_pred in self.hardware_graph.predecessors(pred_id):
                        if self.hardware_graph.kind_of(view_pred) == "data":
                            if grouped_data_to_concat and view_pred in grouped_data_to_concat:
                                cid = grouped_data_to_concat[view_pred]
                                if cid in data_nodes and cid not in input_data_nodes:
                                    input_data_nodes.append(cid)
                            elif view_pred in data_nodes:
                                input_data_nodes.append(view_pred)
                if self.hardware_graph.kind_of(pred_id) == "concat":
                    # 处理concat节点的输入
                    if pred_id in data_nodes:
                        input_data_nodes.append(pred_id) 
                        
            for succ_id in self.hardware_graph.successors(op_node_id):
                if self.hardware_graph.kind_of(succ_id) == "data":
                    if grouped_data_to_concat and succ_id in grouped_data_to_concat:
                        cid = grouped_data_to_concat[succ_id]
                        if cid in data_nodes and cid not in output_data_nodes:
                            output_data_nodes.append(cid)
                    elif succ_id in data_nodes:
                        output_data_nodes.append(succ_id)
            
            # 更新依赖关系
            for input_node in input_data_nodes:
                dependency_graph[input_node]['last_use'] = op_node_id
                
                for output_node in output_data_nodes:
                    # 输入依赖输出
                    if input_node == output_node:  # 跳过自依赖
                        continue
                    dependency_graph[output_node]['predecessors'].add(input_node)
                    dependency_graph[input_node]['successors'].add(output_node)
                    
                    # 如果是inplace操作，检查输入是否可以被覆盖
                    if (is_inplace and len(input_data_nodes) == 1 and len(output_data_nodes) == 1 
                        and self._is_node_overwritable(input_node)):
                        dependency_graph[output_node]['can_be_overwritten_by'].add(input_node)
        
        return dependency_graph
    
    def _is_inplace_operation(self, operation: Operation) -> bool:
        """检查操作是否为inplace操作
        
        Args:
            operation: 操作节点
            
        Returns:
            bool: 是否为inplace操作
        """
        # 检查operation的attrs中是否有inplace参数
        if hasattr(operation, 'attrs') and operation.attrs:
            return operation.attrs.get('inplace', False)
        return False
    
    def _get_node_memory_size(self, node_id: str) -> int:
        """获取节点的内存大小
        
        Args:
            node_id: 节点ID
            
        Returns:
            int: 内存大小（以地址单位计算），如果没有memref则返回0
        """
        node = self.hardware_graph.node(node_id)
        if isinstance(node, Data) and node.memref:
            return node.memref.length
        # ConcatData 的大小来源于其 inferred_memref.length
        if isinstance(node, ConcatData) and getattr(node, 'inferred_memref', None):
            if node.inferred_memref is not None and getattr(node.inferred_memref, 'length', None) is not None:
                return node.inferred_memref.length
            return 0
        else:
            # 没有memref的节点记录下来，不估算大小
            if node_id not in self.nodes_without_memref:
                self.nodes_without_memref.append(node_id)
            return 0
    
    def _is_node_overwritable(self, node_id: str) -> bool:
        """检查节点是否可以被覆盖
        
        Args:
            node_id: 节点ID
            
        Returns:
            bool: 是否可以被覆盖
        """
        # 检查节点名称是否匹配任何不可覆盖模式
        for pattern in self.non_overwritable_patterns:
            if re.search(pattern, node_id):
                return False
        return True
    
    def _topological_sort(self, dependency_graph: Dict[str, Dict[str, Any]]) -> List[str]:
        """对依赖图进行拓扑排序
        
        Args:
            dependency_graph: 依赖图
            
        Returns:
            List[str]: 拓扑排序后的节点列表
        """
        # 计算入度
        in_degree = {}
        for node_id in dependency_graph:
            in_degree[node_id] = len(dependency_graph[node_id]['predecessors'])
        
        # 找到所有入度为0的节点，优先处理有memref的节点
        queue = []
        no_memref_queue = []
        result = []
        
        for node_id, degree in in_degree.items():
            if degree == 0:
                if dependency_graph[node_id]['has_memref']:
                    queue.append(node_id)
                else:
                    no_memref_queue.append(node_id)
            if not self._is_node_overwritable(node_id):
                result.append(node_id)
        
        # # 将没有memref的节点放到队列末尾
        # queue.extend(no_memref_queue)
        
        
        while queue:
            # 选择入度为0的节点
            current = queue.pop(0)
            if not current in result:
                result.append(current)
            
            # 更新后继节点的入度
            for successor in dependency_graph[current]['successors']:
                in_degree[successor] -= 1
                # if in_degree[successor] == 0 and successor not in result and successor not in queue and successor not in no_memref_queue:
                if in_degree[successor] == 0:
                    if not (successor not in queue and successor not in no_memref_queue):
                        warnings.warn(f"Memory Allocator: 节点 {successor} 已在队列中，跳过重复添加")
                        continue

                    if dependency_graph[successor]['has_memref']:
                        queue.append(successor)
                    else:
                        no_memref_queue.append(successor)

        # 检查是否有环
        while True:
            if len(result) != len(dependency_graph):
                warnings.warn(f"Memory Allocator: 应有 {len(dependency_graph)} 个节点，实际排序得到 {len(result)} 个节点，可能存在环路或未处理节点")
                if len(result) < len(dependency_graph):
                    missing_nodes = set(dependency_graph.keys()) - set(result)
                    warnings.warn(f"Memory Allocator: 缺失的节点包括: {missing_nodes}")
                
                # 找到所有环
                cycles = []
                visited = set()
                rec_stack = set()
                def dfs(node_id: str, path: List[str]):
                    visited.add(node_id)
                    rec_stack.add(node_id)
                    path.append(node_id)
                    
                    for successor in dependency_graph[node_id]['successors']:
                        if successor not in visited:
                            dfs(successor, path)
                        elif successor in rec_stack:
                            # 找到环
                            cycle_start_index = path.index(successor)
                            cycle = path[cycle_start_index:]
                            cycles.append(cycle)
                    
                    rec_stack.remove(node_id)
                    path.pop()
                for node_id in dependency_graph:
                    if node_id not in visited:
                        dfs(node_id, [])
                for cycle in cycles:
                    warnings.warn(f"Memory Allocator: 发现环路: {' -> '.join(cycle)}")
                queue.append(cycles[0][0])
            else:
                break
            
            while queue:
                current = queue.pop(0)
                if not current in result:
                    result.append(current)
                
                # 更新后继节点的入度
                for successor in dependency_graph[current]['successors']:
                    in_degree[successor] -= 1
                    if in_degree[successor] == 0:
                        if not (successor not in queue and successor not in no_memref_queue):
                            warnings.warn(f"Memory Allocator: 节点 {successor} 已在队列中，跳过重复添加")
                            continue

                        if dependency_graph[successor]['has_memref']:
                            queue.append(successor)
                        else:
                            no_memref_queue.append(successor)
        return result
    
    def _allocate_node_memory(
        self, 
        node_id: str, 
        core_id: Tuple[int, int], 
        dependency_graph: Dict[str, Dict[str, Any]],
        existing_addresses: Dict[str, int] = None
    ) -> int:
        """为单个节点分配内存
        
        Args:
            node_id: 节点ID
            core_id: 核ID
            dependency_graph: 依赖图
            existing_addresses: 已分配地址映射

        Returns:
            int: 分配的起始地址
            
        Raises:
            MemoryAllocationError: 当内存不足时
        """
        node_info = dependency_graph[node_id]
        required_size = node_info['size']
        
        # TODO: inplace操作优化
        
        # 这类应该每次根据前驱构建一个虚拟的地址空间，然后在其中分配
        # 1. 分析前驱节点：找出该节点的所有依赖路径上的前驱节点
        # 这些节点的状态决定了当前分配的时间步以及哪些内存可以回收
        all_predecessors = set()
        if self._is_node_overwritable(node_id):
            def dfs(current_node: str):
                for pred in dependency_graph[current_node]['predecessors']:
                    if pred not in all_predecessors:
                        all_predecessors.add(pred)
                        dfs(pred)
            dfs(node_id)
        
        # 2. 确定当前分配的时间步
        # 基于前驱节点的分配时间，计算当前节点的逻辑时间步（用于可视化生命周期）
        current_time_step = 0
        max_time_step = 0
        if self._is_node_overwritable(node_id):
            for pred in dependency_graph[node_id]['predecessors']:
                # dependency_graph中的所有节点都在一个核上，所以不需要检查core_id一致性
                assert pred in self.lifecycle_records[core_id], f"前驱节点 {pred} 的生命周期记录不存在"
                assert self.lifecycle_records[core_id][pred]['free_time'] is None, f"前驱节点 {pred} 不应该被释放"
                assert self.lifecycle_records[core_id][pred]['allocate_time'] is not None, f"前驱节点 {pred} 未分配地址"
            for pred in all_predecessors:
                pred_time = self.lifecycle_records[core_id][pred]['allocate_time']
                if pred_time is not None and pred_time > max_time_step:
                    max_time_step = pred_time
            current_time_step = max_time_step + 1
        else:
            current_time_step = 0
        
        # 3. 构建当前时刻的虚拟内存状态
        # 模拟内存分配，决定哪些已分配的块需要保持占用，哪些可以被复用
        vm = VirtualMemory(total_size=self.core_memory_info[core_id]['total_memory'])
        for nid in self.core_allocations[core_id]:
            if nid in all_predecessors:
                # 如果是前驱节点：
                assert nid in self.core_allocations[core_id], f"前驱节点 {nid} 的分配记录不存在"
                # 检查只有当前节点还需要使用该前驱，还是后续其他节点也需要使用
                if self._will_be_used_later(nid, dependency_graph, all_predecessors):
                    # 如果后续还会用到，则该内存块必须保持占用，不能覆盖
                    # 这里的逻辑是，只有当该前驱节点（不一定是直接前驱）的所有直接后继节点也是当前节点的前驱节点时
                    # 说明当前节点出现时，该前驱节点一定已经使用完毕，才能释放
                    start_addr, end_addr = self.core_allocations[core_id][nid]
                    vm.allocate_wo_check(start_addr, end_addr - start_addr + 1)
                else:
                    # 如果该前驱仅被当前链路使用完毕，且后续不再使用，则不将其标记为占用
                    # 这意味着这块内存可以在 VirtualMemory 中被视为"空闲"，从而实现复用
                    if self.lifecycle_records[core_id][nid]['free_time'] is None:
                        self._record_lifecycle_event(nid, 'free', time_step=current_time_step)
            else:
                # 非前驱节点（如其他无关分支的数据），必须保持占用，防止冲突
                start_addr, end_addr = self.core_allocations[core_id][nid]
                vm.allocate_wo_check(start_addr, end_addr - start_addr + 1)
        
        # 4. 执行分配
        if existing_addresses and node_id in existing_addresses:
            # 增量分配或预设地址：尝试在指定位置分配
            if not vm.try_allocate_at(existing_addresses[node_id], required_size):
                raise MemoryAllocationError(
                    f"核 {core_id} 上内存不足，无法为节点 {node_id} 分配*已有*地址 {existing_addresses[node_id]} 大小 {required_size} 的内存"
                )
            address = existing_addresses[node_id]
        else:
            # 新分配：寻找合适的空闲块
            allocation_result = vm.try_allocate(required_size)
            if allocation_result is None:
                raise MemoryAllocationError(
                    f"核 {core_id} 上内存不足，无法为节点 {node_id} 分配 {required_size} 单位的内存"
                )
            address = allocation_result
        
        # 5. 更新记录
        # 记录分配信息
        self.core_allocations[core_id][node_id] = (address, address + required_size - 1)
        
        # 记录分配事件
        self._record_lifecycle_event(node_id, 'allocate', addr=address, size=required_size, time_step=current_time_step)

        return address
    
    def _will_be_used_later(self, node_id: str, dependency_graph: Dict[str, Dict[str, Any]], used_pool) -> bool:
        """检查节点是否还会被后续操作使用
        
        Args:
            node_id: 节点ID
            dependency_graph: 依赖图
            
        Returns:
            bool: 是否还会被使用
        """
        if not self._is_node_overwritable(node_id):
            return True
        if node_id not in dependency_graph:
            raise ValueError(f"节点 {node_id} 不在依赖图中")
        if len(dependency_graph[node_id]['successors']) == 0:
            return False
        
        # 检查所有后继节点是否都已分配
        # core_id = self.hardware_graph.get_core_id(node_id)
        # successor_all_allocated = True
        # for successor in dependency_graph[node_id]['successors']:
        #     if core_id != self.hardware_graph.get_core_id(successor):
        #         continue
        #     if not successor in self.core_allocations[core_id]:
        #         successor_all_allocated = False
        #         break
        successor_all_allocated = True
        for successor in dependency_graph[node_id]['successors']:
            if successor not in used_pool:
                successor_all_allocated = False
                break
                # if len(dependency_graph[successor]['successors']) > 0:
                #     successor_all_allocated = False
                #     break        
        return not successor_all_allocated    
    
    def _update_memref_addresses(self, address_mapping: Dict[str, int]):
        """更新hardware_graph中所有data节点的memref地址
        
        Args:
            address_mapping: 节点名称到地址的映射
        """
        for node_id, address in address_mapping.items():
            node = self.hardware_graph.node(node_id)
            if isinstance(node, Data) and node.memref:
                # 更新memref的地址
                node.memref.addr = address
            elif isinstance(node, ConcatData):
                if node.inferred_memref is None:
                    node.inferred_memref = MemBlock(length=node.memref.length if node.memref else 0, addr=address)
                else:
                    node.inferred_memref.addr = address
                inputs = self.hardware_graph.input_pairs(node_id)  # Dict[int, Data]
                node.allocate_memref(inputs)
        self.hardware_graph.update_viewdata_memrefs()
    
    def get_memory_statistics(self) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """获取内存分配统计信息
        
        Returns:
            Dict[Tuple[int, int], Dict[str, Any]]: 每个核的内存使用统计
        """
        statistics = {}
        
        for core_id, alloc_info in self.core_allocations.items():
            total_allocated = sum(
                end - start + 1 
                for start, end, _ in alloc_info['allocated_blocks']
            )
            total_free = sum(
                end - start + 1 
                for start, end in alloc_info['free_blocks']
            )
            
            statistics[core_id] = {
                'total_memory': alloc_info['total_memory'],
                'allocated_memory': total_allocated,
                'free_memory': total_free,
                'utilization': total_allocated / alloc_info['total_memory'],
                'allocated_blocks': alloc_info['allocated_blocks'].copy(),
                'free_blocks': alloc_info['free_blocks'].copy()
            }
        
        return statistics
    
    def print_allocation_summary(self):
        """打印内存分配摘要"""
        print("\n=== 内存分配摘要 ===")
        
        statistics = self.get_memory_statistics()
        
        for core_id, stats in statistics.items():
            print(f"\n核 {core_id}:")
            print(f"  总内存: {stats['total_memory']}")
            print(f"  预留空间: {stats['reserved_space']}")
            print(f"  已分配: {stats['allocated_memory']}")
            print(f"  空闲: {stats['free_memory']}")
            print(f"  利用率: {stats['utilization']:.2%}")
            
            if stats['allocated_blocks']:
                print("  已分配块:")
                for start, end, node_id in stats['allocated_blocks']:
                    size = end - start + 1
                    print(f"    {node_id}: [{start}:{end}] (大小: {size})")
            
            if stats['free_blocks']:
                print("  空闲块:")
                for start, end in stats['free_blocks']:
                    size = end - start + 1
                    print(f"    [{start}:{end}] (大小: {size})")

