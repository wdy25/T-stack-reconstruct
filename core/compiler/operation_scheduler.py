"""
OperationScheduler
===================

调度器职责:
1. 读取硬件图 (HardwareGraph)
2. 为每个核心生成该核心需执行的 Operation / CommOp / ControlOp 顺序列表
   - 顺序需满足: 节点之间的数据依赖 + 控制依赖
   - 仅输出真正可执行的原语: Operation / CommOp / ControlOp
     (Data / ViewData 只是依赖媒介, 不放入执行列表)
3. 对于每个核心上的线性序列 L = [p0, p1, p2, ...] 中的第 i 个原语 pi, 生成其依赖关系
   相对于它之前的 n 条原语 (n 可由参数 max_backtrack 控制, 默认全部) 的依赖列表:
      deps(pi) = [ j | 0 <= j < i, pj 是 pi 的(直接或间接)前驱 且 i-j <= max_backtrack ]
   这里输出形式可包含:
      - predecessor index j
      - 依赖类型: data / control / cross_core_data (区分是否通过 Send/Recv 来的)
      - 依赖的具体数据节点集合 (可选)

设计说明:
---------
A. 获取 per-core 节点集合: hardware_graph.get_nodes_by_core(core_id)
B. 抽取其中可执行原语节点 (operation / communication / control)
C. 利用 Kahn 拓扑排序, 但需要限制在该核心节点上, 同时依赖的前驱如果在其他核心, 需要通过 Send/Recv 节点已经被插入, 故直接视为依赖.
D. 生成序列后, 再做一次回溯, 建立局部索引的依赖关系.

返回结构:
---------
SchedulerResult = {
  core_id: {
     'op_list': [node_id0, node_id1, ...],
     'dependencies': [  # 与 op_list 对应
        [ {'src_index': j, 'type': 'data'|'control'|'cross_core', 'via': <edge data nodes list>} , ...],
        ...
     ]
  },
  ...
}

后续扩展: 可加入调度策略(优先级, latency 模型, 资源约束等)。
"""
from __future__ import annotations
from myhdl import bin, intbv
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Set
from core.ir.hardware_graph import HardwareGraph, NodeId
from core.ir.graph import Graph
from core.ir.operation import Operation
from core.ir.control_op import ControlOp
from core.ir.communication_op import CommOp, SendOp, RecvOp
from core.ir.operations.loop import Loop

ExecutableKind = ("operation", "control", "communication")


class OperationScheduler:
    def __init__(self, hardware_graph: HardwareGraph):
        self.hw_graph = hardware_graph
        self.op_list: Dict[Tuple[int,int], List[NodeId]] = {}
        self.predecessors: Dict[Tuple[int,int], Dict[NodeId, Set[NodeId]]] = {}
        self.successors: Dict[Tuple[int,int], Dict[NodeId, Set[NodeId]]] = {}
        self.executable_nodes_cache: Dict[Tuple[int,int], List[NodeId]] = {}
        self.cfgs: Dict[Tuple[int,int], Any] = {}

    def _is_executable(self, nid: NodeId) -> bool:
        kind = self.hw_graph.kind_of(nid)
        return kind in ExecutableKind

    def build_core_op_lists(self, try_parallel=False, remove_deadlock=False) -> Dict[Tuple[int,int], List[NodeId]]:
        """为每个核心生成顺序列表与依赖(完整依赖, 回溯不限)."""
        for core_id in self.hw_graph.get_all_core_ids():
            schedule = self._build_single_core_schedule(core_id, try_parallel=try_parallel)
            self.op_list[core_id] = schedule
        
        if remove_deadlock:
            self._remove_deadlock(try_parallel)

        return self.op_list

    def _remove_deadlock(self, try_parallel=False):
        """
        去除跨核通信可能导致的死锁。
        策略：
        1. 识别成对的 Send/Recv 操作。
        2. 构建通信操作的全局依赖图（Link Dependency Graph）。
           - 节点是 (Send, Recv) 对。
           - 边表示必须的执行顺序，来源于核内的依赖关系。
        3. 对 Link Graph 进行拓扑排序，得到全局一致的通信顺序。
        4. 将此顺序作为新的依赖约束加回到各核的依赖图中。
        5. 重新调度。
        """
        # 1. 收集所有 Send/Recv 操作
        send_ops = []
        recv_ops = []
        node_to_core = {}
        
        for core_id, ops in self.op_list.items():
            for nid in ops:
                node_to_core[nid] = core_id
                node = self.hw_graph.node(nid)
                if isinstance(node, SendOp):
                    send_ops.append(nid)
                elif isinstance(node, RecvOp):
                    recv_ops.append(nid)
        
        # 2. 匹配 Send/Recv 对
        # Key: (src_core, dst_core, tag)
        sends_by_key = {}
        for nid in send_ops:
            node = self.hw_graph.node(nid)
            src = node_to_core[nid]
            assert(len(node.msg_list) == 1), "只支持单消息的 SendOp"
            msg = node.msg_list[0]
            assert msg.Q == 0, "只支持 Q=0 的 SendMsg"
            dst = (src[0] + msg.Y, src[1] + msg.X)
            if isinstance(src, list): src = tuple(src)
            if isinstance(dst, list): dst = tuple(dst)
            tag = msg.tag_id
            key = (dst, tag)
            if key not in sends_by_key: sends_by_key[key] = []
            else: assert(0), "不支持同一 SendOp 多次发送相同 tag"
            sends_by_key[key].append(nid)

        recvs_by_key = {}
        for nid in recv_ops:
            node = self.hw_graph.node(nid)
            dst = node_to_core[nid]
            # src = node.attrs['source']
            # if isinstance(src, list): src = tuple(src)
            tag = node.attrs['tag']
            key = (dst, tag)
            if key not in recvs_by_key: recvs_by_key[key] = []
            else: assert(0), "不支持同一 RecvOp 多次接收相同 tag"
            recvs_by_key[key].append(nid)
            
        links = [] # List of (send_node, recv_node)
        all_keys = set(sends_by_key.keys()) | set(recvs_by_key.keys())
        
        for key in all_keys:
            s_list = sends_by_key.get(key, [])
            r_list = recvs_by_key.get(key, [])
            
            if len(s_list) != len(r_list):
                warnings.warn(f"Unmatched Send/Recv for key {key}: {len(s_list)} sends, {len(r_list)} recvs")
            
            # 按当前调度顺序排序，作为默认匹配顺序
            src_core = node_to_core[s_list[0]]
            dst_core = key[0]
            
            # 确保 core_id 存在于 op_list 中 (防止配置错误导致 key 中的 core_id 无效)
            if src_core in self.op_list:
                s_list.sort(key=lambda nid: self.op_list[src_core].index(nid))
            if dst_core in self.op_list:
                r_list.sort(key=lambda nid: self.op_list[dst_core].index(nid))
            
            n = min(len(s_list), len(r_list))
            for i in range(n):
                links.append((s_list[i], r_list[i]))

        if not links:
            return

        # 3. 构建 Link Dependency Graph
        adj = {i: set() for i in range(len(links))}
        in_degree = {i: 0 for i in range(len(links))}
        
        memo_reachable = {}
        def get_reachable(u, core_id):
            if u in memo_reachable: return memo_reachable[u]
            visited = set()
            stack = [u]
            while stack:
                curr = stack.pop()
                # 使用 self.successors 获取依赖后继
                if core_id in self.successors and curr in self.successors[core_id]:
                    for succ in self.successors[core_id][curr]:
                        if succ not in visited:
                            visited.add(succ)
                            stack.append(succ)
            memo_reachable[u] = visited
            return visited

        for i in range(len(links)):
            s1, r1 = links[i]
            for j in range(len(links)):
                if i == j: continue
                s2, r2 = links[j]
                
                # 检查 Sender 侧依赖
                if node_to_core[s1] == node_to_core[s2]:
                    core = node_to_core[s1]
                    if s2 in get_reachable(s1, core): adj[i].add(j)
                    elif s1 in get_reachable(s2, core): adj[j].add(i)
                
                # 检查 Receiver 侧依赖
                if node_to_core[r1] == node_to_core[r2]:
                    core = node_to_core[r1]
                    if r2 in get_reachable(r1, core): adj[i].add(j)
                    elif r1 in get_reachable(r2, core): adj[j].add(i)
                
                # 检查混合依赖 (同一核上的 Send 和 Recv)
                if node_to_core[s1] == node_to_core[r2]:
                    core = node_to_core[s1]
                    if r2 in get_reachable(s1, core): adj[i].add(j)
                    elif s1 in get_reachable(r2, core): adj[j].add(i)
                
                if node_to_core[r1] == node_to_core[s2]:
                    core = node_to_core[r1]
                    if s2 in get_reachable(r1, core): adj[i].add(j)
                    elif r1 in get_reachable(s2, core): adj[j].add(i)

        # 4. 拓扑排序
        for u in adj:
            for v in adj[u]:
                in_degree[v] += 1
        
        queue = [i for i in range(len(links)) if in_degree[i] == 0]
        sorted_links_indices = []
        
        while queue:
            u = queue.pop(0)
            sorted_links_indices.append(u)
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        if len(sorted_links_indices) < len(links):
            raise RuntimeError("Deadlock detected: Cycle in communication dependencies.")
            
        # 5. 应用新约束并重新调度
        changes_made = False
        
        # 提取每个核上的通信操作的新顺序
        ops_by_core_in_order = {}
        for idx in sorted_links_indices:
            s, r = links[idx]
            c_s = node_to_core[s]
            if c_s not in ops_by_core_in_order: ops_by_core_in_order[c_s] = []
            ops_by_core_in_order[c_s].append(s)
            
            c_r = node_to_core[r]
            if c_r not in ops_by_core_in_order: ops_by_core_in_order[c_r] = []
            ops_by_core_in_order[c_r].append(r)
            
        for core_id, ops in ops_by_core_in_order.items():
            for k in range(len(ops) - 1):
                u, v = ops[k], ops[k+1]
                # 添加依赖 u -> v
                if core_id not in self.predecessors: self.predecessors[core_id] = {}
                if core_id not in self.successors: self.successors[core_id] = {}
                
                if v not in self.predecessors[core_id]: self.predecessors[core_id][v] = set()
                if u not in self.successors[core_id]: self.successors[core_id][u] = set()
                
                if u not in self.predecessors[core_id][v]:
                    self.predecessors[core_id][v].add(u)
                    self.successors[core_id][u].add(v)
                    changes_made = True

        if changes_made:
            print("Deadlock removal: Added constraints and rescheduling...")
            for core_id in self.hw_graph.get_all_core_ids():
                old_schedule = list(self.op_list[core_id])
                new_schedule = self._build_single_core_schedule(core_id, try_parallel=try_parallel, rebuild_deps=False)
                self.op_list[core_id] = new_schedule
                
                if old_schedule != new_schedule:
                    print(f"Core {core_id} schedule changed.")
        else:
            print("Deadlock removal: No changes needed.")
    
    def _build_dependency_graph(self, core_id: Tuple[int,int]):
         # 1. 收集该核心上的可执行节点
        core_nodes = set(self.hw_graph.get_nodes_by_core(core_id))
        executable_nodes = [nid for nid in core_nodes if self._is_executable(nid)]
        self.executable_nodes_cache[core_id] = executable_nodes
        # 构建局部入度: 只考虑 data/control 前驱里也是 executable 并且在同核的情况
        in_degree: Dict[NodeId, int] = {}
        predecessors_cache: Dict[NodeId, Set[NodeId]] = {}
        successors_cache: Dict[NodeId, Set[NodeId]] = {}
        for nid in executable_nodes:
            if nid not in successors_cache:
                successors_cache[nid] = set()
            if nid not in predecessors_cache:
                predecessors_cache[nid] = set()
             # 收集数据前驱 (考虑 view/concat 展开)
            data_preds = [p for p in self.hw_graph.predecessors(nid) if self.hw_graph.kind_of(p) in ("data",)]
            for p in self.hw_graph.predecessors(nid):
                if self.hw_graph.kind_of(p) in ("view",):
                    assert len(self.hw_graph.predecessors(p)) == 1, "ViewData should have exactly one predecessor"
                    data_preds.append(self.hw_graph.predecessors(p)[0])
                if self.hw_graph.kind_of(p) in ("concat",):
                    for q in self.hw_graph.predecessors(p):
                        if q in core_nodes:
                            assert self.hw_graph.kind_of(q) in ("data")
                            data_preds.append(q)
                    
            ctrl_preds = self.hw_graph.control_predecessors(nid)
            # 其可执行前驱: 数据前驱的生产者(如果在本核且为可执行), 控制前驱(如果在本核)
            exec_preds: Set[NodeId] = set()
            # 数据前驱 -> 找到产生该数据的 operation/comm (唯一或0个). 如果数据节点在本核, 查看其单前驱
            for d in data_preds:
                prod_list = self.hw_graph.predecessors(d)
                for prod in prod_list:
                    if prod in core_nodes and self._is_executable(prod) and prod != nid:
                        exec_preds.add(prod)
            for c in ctrl_preds:
                if c in core_nodes and self._is_executable(c) and c != nid:
                    exec_preds.add(c)
            for p in exec_preds:
                if p not in successors_cache:
                    successors_cache[p] = set()
                if p != nid: # 防止自环
                    successors_cache[p].add(nid)
            predecessors_cache[nid] = exec_preds
        self.predecessors[core_id] = predecessors_cache
        self.successors[core_id] = successors_cache
    
    def _build_single_core_schedule(self, core_id: Tuple[int,int], try_parallel=False, rebuild_deps=True) -> List[NodeId]:
        if rebuild_deps:
            self._build_dependency_graph(core_id)
        self._build_single_core_cfg(core_id)
        
        # 检验cfg中blk之间的连通性
        blocks = self.cfgs[core_id]
        named_blks = {i: blk for i, blk in enumerate(blocks)}
        blk_predecessors: Dict[int, Set[int]] = {i: set() for i in named_blks.keys()}
        blk_successors: Dict[int, Set[int]] = {i: set() for i in named_blks.keys()}
        for i, blk in named_blks.items():
            if blk.control_entry is not None:
                for j, other_blk in named_blks.items():
                    if i != j and blk.control_entry == other_blk.control_exit:
                        blk_predecessors[i].add(j)
                        blk_successors[j].add(i)
            if blk.control_exit is not None:
                for j, other_blk in named_blks.items():
                    if i != j and blk.control_exit == other_blk.control_entry:
                        blk_successors[i].add(j)
                        blk_predecessors[j].add(i)
        
        ordered = []
        
        # 寻找遍历的起点
        # 优先级1: 没有 control 前驱的 block
        start_blocks = [i for i, preds in blk_predecessors.items() if len(preds) == 0]
        if not start_blocks:
            # 优先级2: control_entry是loop且有control_exit的block
            for i, blk in named_blks.items():
                if blk.control_entry is not None:
                    ctrl_node = self.hw_graph.node(blk.control_entry)
                    if isinstance(ctrl_node, Loop) and blk.control_exit is not None:
                        # blk.entry_nodes只有一个且必须是control_entry的第二个port
                        assert len(blk.entry_nodes) == 1
                        if self.hw_graph.control_output_pairs(blk.control_entry)[1].name == blk.entry_nodes[0]:
                            start_blocks.append(i)
            if not start_blocks:
                # 优先级3：control_entry是任意ControlOp且有control_exit的block
                for i, blk in named_blks.items():
                    if blk.control_entry is not None:
                        ctrl_node = self.hw_graph.node(blk.control_entry)
                        if isinstance(ctrl_node, ControlOp) and blk.control_exit is not None:
                            # blk.entry_nodes只有一个且必须是control_entry的第二个port
                            assert len(blk.entry_nodes) == 1
                            if self.hw_graph.control_output_pairs(blk.control_entry)[1].name == blk.entry_nodes[0]:
                                start_blocks.append(i)
        if not start_blocks:
            raise RuntimeError("Cannot find a valid starting block for scheduling.")
        
        visited_blocks = set()
        while start_blocks:
            current_blk_id = start_blocks.pop(0)
            if current_blk_id in visited_blocks:
                continue
            visited_blocks.add(current_blk_id)
            current_blk = named_blks[current_blk_id]

            # 对当前 block 内部进行拓扑排序
            # 构建局部入度
            in_degree_blk: Dict[NodeId, int] = {nid: len(current_blk.predecessors[nid]) for nid in current_blk.nodes}
            # Kahn
            queue_blk: List[NodeId] = [nid for nid in current_blk.nodes if in_degree_blk[nid] == 0]
            ordered_blk: List[NodeId] = []
            while queue_blk:
                if not try_parallel:
                    u = queue_blk.pop(0)
                else:  # 优先选择和上一个节点没有依赖关系的节点
                    for idx, candidate in enumerate(queue_blk):
                        if len(ordered_blk) == 0 or candidate not in self.predecessors[core_id][ordered_blk[-1]]:
                            u = candidate
                            del queue_blk[idx]
                            break
                    else:
                        u = queue_blk.pop(0)
                ordered_blk.append(u)
                # 找所有以 u 为前驱(数据或控制)的可执行节点
                for v in current_blk.successors[u]:
                    if u == v:
                        continue
                    in_degree_blk[v] -= 1
                    if in_degree_blk[v] == 0:
                        queue_blk.append(v)
            # 若有环(理论上不应发生), 直接按剩余追加
            if len(ordered_blk) < len(current_blk.nodes):
                warnings.warn("Detected a cycle in the block, appending remaining nodes.")
                leftover = [nid for nid in current_blk.nodes if nid not in ordered_blk]
                ordered_blk.extend(leftover)
            ordered.extend(ordered_blk)
            
            # 将后继 block 加入待处理队列
            if current_blk.control_exit is None:
                continue
            if not current_blk.control_exit in ordered:
                ordered.append(current_blk.control_exit)
            control_outputs = self.hw_graph.control_output_pairs(current_blk.control_exit)
            assert len(control_outputs) <= 2, "Control exit should have at most 2 control outputs."
            # 优先处理port 0的后继
            for port in range(len(control_outputs)):
                succ_node = control_outputs[port].name
                for j, other_blk in named_blks.items():
                    if current_blk.control_exit == other_blk.control_entry and j not in visited_blocks and succ_node in other_blk.entry_nodes:
                        # 如果输出有两个且目前处理的是port 0，则将port 0放在队列开头（之后马上处理）
                        if port == 0 and len(control_outputs) == 2:
                            start_blocks.insert(0, j)
                        else:
                            start_blocks.append(j)
        
        if len(ordered) < len(self.executable_nodes_cache[core_id]):
            warnings.warn("Not all executable nodes were scheduled, appending remaining nodes.")
            leftover = [nid for nid in self.executable_nodes_cache[core_id] if nid not in ordered]
            print(leftover)
            ordered.extend(leftover)

        return ordered
    
    def build_deps_for_ops(self, max_backtrack: int = 8) -> Dict[Tuple[int,int], List[intbv]]:
        """为每个核心的每个原语生成依赖列表, 回溯不超过 max_backtrack."""
        all_deps: Dict[Tuple[int,int], List[intbv]] = {}
        for core_id, op_list in self.op_list.items():
            core_deps: List[intbv] = []
            preds_cache = self.predecessors[core_id]
            nid_to_index = {nid: idx for idx, nid in enumerate(op_list)}
            for i, nid in enumerate(op_list):
                backtrack_limit = max_backtrack if max_backtrack is not None else i
                deps: intbv = intbv(0)[backtrack_limit:]  # 使用位向量表示依赖关系
                for j in range(max(0, i - backtrack_limit), i):
                    pred_nid = op_list[j]
                    print(deps)
                    if pred_nid in preds_cache[nid]:
                        deps[i-j-1] = 1
                core_deps.append(deps)
            all_deps[core_id] = core_deps
        
        return all_deps

    def _build_single_core_cfg(self, core_id: Tuple[int,int]):
        """基于 per-core schedule 构造控制流图(CFG)的 block 列表。
        规则:
        1. block 只包含非 control 节点。
        2. 与 control_op 的 control 连接视为 block 的进入或退出点。
        3. 无任何可执行前驱的算子视为天然进入点。
        4. 每个 block 仅允许一个基于 control 的进入/退出点；若不存在则可有多个天然进入/退出点。
        返回结构:
        { core_id: { 'blocks': [ { 'id': int, 'nodes': [...], 'entry': { 'control': NodeId|None, 'natural': [...] }, 'exit': { 'control': NodeId|None, 'natural': [...] } , 'succ_blocks': [ids], 'pred_blocks': [ids] } ] } }
        """
        
        class Block:
            def __init__(self, block_id: int):
                self.control_entry: Optional[NodeId] = None
                self.control_exit: Optional[NodeId] = None
                self.entry_nodes: List[NodeId] = []
                self.exit_nodes: List[NodeId] = []
                self.nodes: List[NodeId] = []
                self.predecessors: Dict[NodeId, Set[NodeId]] = {}
                self.successors: Dict[NodeId, Set[NodeId]] = {}
            
        control_nodes = [nid for nid in self.executable_nodes_cache[core_id] if self.hw_graph.kind_of(nid) == "control"]
        split_edges = set()
        control_entry_nodes: Dict[NodeId, NodeId] = {}
        control_exit_nodes: Dict[NodeId, NodeId] = {}
        for cnode in control_nodes:
            for succ in self.hw_graph.control_successors(cnode):
                if succ in self.executable_nodes_cache[core_id]:
                    control_entry_nodes[succ] = cnode
                    for p in self.predecessors[core_id][succ]:
                        split_edges.add((p, succ))
            for pred in self.hw_graph.control_predecessors(cnode):
                if pred in self.executable_nodes_cache[core_id]:
                    control_exit_nodes[pred] = cnode
                    for s in self.successors[core_id][pred]:
                        split_edges.add((pred, s))
        
        visited = set()
        blocks: List[Block] = []
        def dfs(node: NodeId, blk: Block) -> None:
            if node in visited:
                return
            visited.add(node)
            blk.nodes.append(node)
            blk.predecessors[node] = set()
            blk.successors[node] = set()
            
            if node in control_entry_nodes:
                assert blk.control_entry is None, "Block cannot have multiple control entry points"
                blk.control_entry = control_entry_nodes[node]
                blk.entry_nodes.append(node)
                assert len(blk.entry_nodes) == 1, "Block cannot have multiple control entry points"
            else:
                # 天然入口
                if all((pred, node) in split_edges for pred in self.predecessors[core_id][node]) or len(self.predecessors[core_id][node]) == 0:
                    assert blk.control_entry is None, "Block cannot have multiple natural entry points"
                    blk.entry_nodes.append(node)
            
            if node in control_exit_nodes:
                assert blk.control_exit is None, "Block cannot have multiple control exit points"
                blk.control_exit = control_exit_nodes[node]
                blk.exit_nodes.append(node)
                assert len(blk.exit_nodes) == 1, "Block cannot have multiple control exit points"
            else:
                # 天然出口
                if all((node, succ) in split_edges for succ in self.successors[core_id][node]) or len(self.successors[core_id][node]) == 0:
                    assert blk.control_exit is None, "Block cannot have multiple natural exit points"
                    blk.exit_nodes.append(node)
            
            # 遍历依赖连接（忽略分割边）
            for succ in self.successors[core_id][node]:
                if (node, succ) not in split_edges:
                    blk.successors[node].add(succ)
                    if succ not in blk.predecessors:
                        blk.predecessors[succ] = set()
                    blk.predecessors[succ].add(node)
                    if succ not in visited:
                        dfs(succ, blk)
            for pred in self.predecessors[core_id][node]:
                if (pred, node) not in split_edges:
                    blk.predecessors[node].add(pred)
                    if pred not in blk.successors:
                        blk.successors[pred] = set()
                    blk.successors[pred].add(node)
                    if pred not in visited:
                        dfs(pred, blk)
        
        for nid in self.executable_nodes_cache[core_id]:
            if nid not in visited and self.hw_graph.kind_of(nid) != "control":
                new_block = Block(len(blocks))
                dfs(nid, new_block)
                blocks.append(new_block)            
        
        self.cfgs[core_id] = blocks
            


__all__ = [
    'OperationScheduler',
]
