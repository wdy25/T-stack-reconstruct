from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import warnings

from core.ir.data import Data, ViewData, ConcatData
from core.ir.operation import Operation
from core.ir.control_op import ControlOp
from core.ir.communication_op import CommOp

NodeId = str

class Graph:
    """DAG with Data and Operation nodes. Supports ported data edges and control edges.

    Port model (minimal, fast):
    - For edges INTO an Operation (Data -> Op), we store the destination port index (dst_port).
    - For edges OUT OF a ComputeOp (Op -> Data), we store the source port index (src_port).
    - Control edges do not use ports and only encode execution order.
    - Ports align with adjacency: _in_ports aligns with _in_edges; _out_ports aligns with _out_edges.
    """

    __slots__ = (
        "_nodes",
        "_in_edges",
        "_out_edges",
        "_in_ports",
        "_out_ports",
        "_ctrl_in_edges",
        "_ctrl_out_edges",
        "_ctrl_in_ports",
        "_ctrl_out_ports",
    )

    def __init__(self) -> None:
        self._nodes: dict[NodeId, Any] = {}
        self._in_edges: dict[NodeId, List[NodeId]] = {}
        self._out_edges: dict[NodeId, List[NodeId]] = {}
        self._in_ports: dict[NodeId, List[Optional[int]]] = {}  # store the input ports of a node
        self._out_ports: dict[NodeId, List[Optional[int]]] = {}  # store the output ports of a node
        self._ctrl_in_edges: dict[NodeId, List[NodeId]] = {}  # store the ctrl flow of a node, most nodes have an empty list
        self._ctrl_out_edges: dict[NodeId, List[NodeId]] = {}  # store the ctrl flow of a node, most nodes have an empty list
        self._ctrl_in_ports: dict[NodeId, List[Optional[int]]] = {}  # store the input ports of a node
        self._ctrl_out_ports: dict[NodeId, List[Optional[int]]] = {}  # store the output ports of a node

    # Node management
    def add_node(self, node: Any) -> NodeId:
        """Add any node (Data or Operation) to the graph."""
        assert node.name not in self._nodes.keys(), f"Node name {node.name} already exists"
        nid = node.name
        self._nodes[nid] = node
        self._in_edges[nid] = []
        self._out_edges[nid] = []
        self._in_ports[nid] = []
        self._out_ports[nid] = []
        self._ctrl_in_edges[nid] = []
        self._ctrl_out_edges[nid] = []
        self._ctrl_in_ports[nid] = []
        self._ctrl_out_ports[nid] = []
        return nid

    def node(self, nid: NodeId) -> Any:
        return self._nodes[nid]

    def all_nodes(self) -> List[NodeId]:
        return list(self._nodes.keys())

    def kind_of(self, nid: NodeId) -> str:
        node = self._nodes[nid]
        # 检查原始类型
        if isinstance(node, ViewData):
            return "view"
        elif isinstance(node, Data):
            return "data"
        elif isinstance(node, Operation):
            return "operation"
        elif isinstance(node, ControlOp):
            return "control"
        elif isinstance(node, CommOp):
            return "communication"
        elif isinstance(node, ConcatData):
            return "concat"
        else:
            raise ValueError(f"Unknown node type: {type(node)}")

    # Data-flow edge management with validation and ports
    def connect(
        self,
        src: NodeId,
        dst: NodeId,
        src_port: Optional[int] = None,
        dst_port: Optional[int] = None,
    ) -> None:
        src_node = self.node(src)
        dst_node = self.node(dst)
        # Allow non-primitive operations to bypass type checks in order to connect intermediate data nodes when unfolding subgraphs.
        # skipe non-primitive ops type checks
        skip_type_checks = (
            isinstance(src_node, Operation) and not getattr(src_node, "primitive", True)
        ) or (
            isinstance(dst_node, Operation) and not getattr(dst_node, "primitive", True)
        )
        # set no skip for now
        # skip_type_checks = False

        src_kind = self.kind_of(src)
        dst_kind = self.kind_of(dst)

        # Enforce ComputeOp data-only IO
        if not skip_type_checks:
            if src_kind == "operation" and dst_kind not in ("data"):
                raise ValueError("Operation outputs must connect to Data or View nodes")
            if dst_kind == "operation" and src_kind not in ("data", "view", "concat"):
                raise ValueError("Operation inputs must come from Data or View nodes")
            
            # View节点的连接规则
            if src_kind == "view":
                # View节点只能作为输出连接到operation
                if not dst_kind in ("operation", "communication", "control"):
                    raise ValueError("View nodes can only connect to Operation nodes")
            
            if dst_kind == "view":
                # View节点只能接收来自data节点的输入
                if src_kind not in ("data", "concat"):
                    raise ValueError("View nodes can only receive input from Data or ConcatData nodes")
            
            # Concat节点的连接规则
            if src_kind == "concat":
                # ConcatData节点不能连到Data节点
                if dst_kind in ("data",):
                    raise ValueError("ConcatData nodes can't connect to Data nodes")

        # Assign default ports
        assigned_src_port = src_port
        assigned_dst_port = dst_port
        if dst_kind in ["operation", "communication", "control"] and src_kind in ("data", "view", "concat") and assigned_dst_port is None:
            for p in self._in_ports[dst]:
                assert p is not None, "Operation input ports must be explicitly assigned"
            used = set(p for p in self._in_ports[dst] if p is not None)
            assigned_dst_port = 0
            while assigned_dst_port in used:
                assigned_dst_port += 1
        if src_kind in ["operation", "communication", "control"] and dst_kind in ("data",) and assigned_src_port is None:
            for p in self._out_ports[src]:
                assert p is not None, "Operation output ports must be explicitly assigned"
            used = set(p for p in self._out_ports[src] if p is not None)
            assigned_src_port = 0
            while assigned_src_port in used:
                assigned_src_port += 1
        
        if src_kind in ("data",) and dst_kind in ("concat",):
            for p in self._in_ports[dst]:
                assert p is not None, "Control input ports must be explicitly assigned"
            used = set(p for p in self._in_ports[dst] if p is not None)
            assigned_dst_port = 0
            while assigned_dst_port in used:
                assigned_dst_port += 1
        # data 和 view 节点的端口始终为 None

        # # Add edge
        # assert src not in self._in_edges[dst], "Edge already exists"
        # assert dst not in self._out_edges[src], "Edge already exists"
        
        self._out_edges[src].append(dst)
        self._in_edges[dst].append(src)
        self._out_ports[src].append(assigned_src_port)
        self._in_ports[dst].append(assigned_dst_port)

        # not cycle check, because of temporal dependency
        # # Cycle check
        # if self._detect_reachable(dst, src):
        #     self._out_edges[src].pop()
        #     self._in_edges[dst].pop()
        #     self._out_ports[src].pop()
        #     self._in_ports[dst].pop()
        #     raise ValueError("Edge creates a cycle; graph must remain a DAG")

    def connect_mapping(
        self,
        src_mapping: List[Tuple[NodeId, int]],
        dst_mapping: List[Tuple[NodeId, int]],
    ) -> None:
        """Fully connect every (node, port) pair in src_mapping to all in dst_mapping."""
        if not src_mapping or not dst_mapping:
            return
        for src_node, src_port in src_mapping:
            if src_node not in self._nodes:
                raise ValueError(f"Source node {src_node} does not exist")
            for dst_node, dst_port in dst_mapping:
                if dst_node not in self._nodes:
                    raise ValueError(f"Destination node {dst_node} does not exist")
                self.connect(src_node, dst_node, src_port=src_port, dst_port=dst_port)

    def _connect_nocheck(
        self,
        src: NodeId,
        dst: NodeId,
        *,
        src_port: Optional[int],
        dst_port: Optional[int],
    ) -> None:
        """Internal: add a DATA edge without validation (used for import).

        Assumes nodes exist and graph remains acyclic; ports are taken as-is.
        """
        self._out_edges[src].append(dst)
        self._in_edges[dst].append(src)
        self._out_ports[src].append(src_port)
        self._in_ports[dst].append(dst_port)

    def connect_control(
        self, 
        src: NodeId, 
        dst: NodeId,
        src_port: Optional[int] = None,
        dst_port: Optional[int] = None,
    ) -> None:
        if self.kind_of(src) in ("data", "view", "concat") or self.kind_of(dst) in ("data", "view", "concat"):
            raise ValueError("Control edges must be between non-Data and non-View nodes")
        self._ctrl_out_edges[src].append(dst)
        self._ctrl_in_edges[dst].append(src)
        
        assigned_src_port = src_port
        assigned_dst_port = dst_port
        
        if assigned_dst_port is None:
            for p in self._ctrl_in_edges[dst]:
                assert p is not None, "Control input ports must be explicitly assigned"
            used = set(p for p in self._ctrl_in_ports[dst] if p is not None)
            assigned_dst_port = 0
            while assigned_dst_port in used:
                assigned_dst_port += 1
        
        if assigned_src_port is None:
            for p in self._ctrl_out_edges[src]:
                assert p is not None, "Control output ports must be explicitly assigned"
            used = set(p for p in self._ctrl_out_ports[src] if p is not None)
            assigned_src_port = 0
            while assigned_src_port in used:
                assigned_src_port += 1
        
        self._ctrl_out_ports[src].append(assigned_src_port)
        self._ctrl_in_ports[dst].append(assigned_dst_port)
        
        # if self._detect_reachable(dst, src):
        #     self._ctrl_out_edges[src].pop()
        #     self._ctrl_in_edges[dst].pop()
        #     raise ValueError("Control edge creates a cycle; graph must remain a DAG")

    # Queries
    def predecessors(self, nid: NodeId) -> List[NodeId]:
        return self._in_edges[nid]

    def successors(self, nid: NodeId) -> List[NodeId]:
        return self._out_edges[nid]

    def control_predecessors(self, nid: NodeId) -> List[NodeId]:
        return self._ctrl_in_edges[nid]

    def control_successors(self, nid: NodeId) -> List[NodeId]:
        return self._ctrl_out_edges[nid]

    def in_ports(self, nid: NodeId) -> List[Optional[int]]:
        return self._in_ports[nid]

    def out_ports(self, nid: NodeId) -> List[Optional[int]]:
        return self._out_ports[nid]
    
    def control_in_ports(self, nid: NodeId) -> List[Optional[int]]:
        return self._ctrl_in_ports[nid]

    def control_out_ports(self, nid: NodeId) -> List[Optional[int]]:
        return self._ctrl_out_ports[nid]

    def nodes(self) -> Iterable[Tuple[NodeId, Any]]:  # type: ignore[name-defined]
        for nid, node in self._nodes.items():
            yield nid, node
            
    def find_cycles(self) -> List[List[NodeId]]:
        """Detect cycles in the graph using DFS.

        Returns:
            List[List[NodeId]]: A list of cycles, each represented as a list of NodeIds.
        """
        visited = set()
        stack = set()
        cycles = []

        def dfs(node: NodeId, path: List[NodeId]) -> None:
            if node in stack:
                cycle_start_index = path.index(node)
                cycles.append(path[cycle_start_index:])
                return
            if node in visited:
                return
            
            visited.add(node)
            stack.add(node)
            path.append(node)

            for neighbor in self._out_edges[node] + self._ctrl_out_edges[node]:
                dfs(neighbor, path)

            stack.remove(node)
            path.pop()

        for nid in self._nodes.keys():
            if nid not in visited:
                dfs(nid, [])

        return cycles

    def topological_order(self) -> List[NodeId]:
        # 标准Kahn算法的拓扑排序，允许存在特定类型的环
        indeg: Dict[NodeId, int] = {}
        for nid in self._nodes.keys():
            indeg[nid] = len(self._in_edges[nid]) + len(self._ctrl_in_edges[nid])

        q: List[NodeId] = [i for (i, d) in indeg.items() if d == 0]
        order: List[NodeId] = []
        processed: set[NodeId] = set()

        def enqueue_if_zero(v: NodeId) -> None:
            if v not in processed and indeg[v] == 0:
                q.append(v)

        def process_node(u: NodeId) -> None:
            processed.add(u)
            order.append(u)
            # 减小所有后继（数据与控制）入度
            for v in self._ctrl_out_edges[u]:
                indeg[v] -= 1
                enqueue_if_zero(v)
            for v in self._out_edges[u]:
                indeg[v] -= 1
                enqueue_if_zero(v)

        # 常规无环部分
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            process_node(u)

        # 存在环：打印环；并对“算子-数据”的两节点环进行成对处理
        if len(processed) != len(self._nodes):
            cycles = self.find_cycles()
            if cycles:
                # 打印环结构
                for cyc in cycles:
                    if not cyc:
                        continue
                    # 格式：A -> B -> C -> A
                    path = " -> ".join(cyc + [cyc[0]])
                    print(f"Detected cycle: {path}")

            remaining: set[NodeId] = set(self._nodes.keys()) - processed

            def process_op_data_pair(a: NodeId, b: NodeId) -> None:
                # 将一对“算子-数据”环作为一个单元处理，保证相邻
                kinds = (self.kind_of(a), self.kind_of(b))
                # 尽量保持数据在前，算子在后
                pair = (a, b)
                if kinds == ("operation", "data"):
                    pair = (b, a)
                elif kinds == ("data", "operation"):
                    pair = (a, b)
                else:
                    # 理论上不会进入（由调用方保证），兜底按原顺序
                    pair = (a, b)

                for u in pair:
                    processed.add(u)
                    order.append(u)

                # 同时“移除”两点：对各自（不含对方）的后继减入度
                for u in pair:
                    for v in self._ctrl_out_edges[u]:
                        if v not in pair:
                            indeg[v] -= 1
                            enqueue_if_zero(v)
                    for v in self._out_edges[u]:
                        if v not in pair:
                            indeg[v] -= 1
                            enqueue_if_zero(v)

            progressed = True
            while progressed and len(processed) != len(self._nodes):
                progressed = False
                # 优先处理所有符合规则的两节点环
                for cyc in cycles:
                    if len(cyc) == 2:
                        n1, n2 = cyc
                        if n1 in processed or n2 in processed:
                            continue
                        # 必须为“数据边”互相连成一环（避免控制边误判）
                        if (n1 in self._out_edges[n2]) and (n2 in self._out_edges[n1]):
                            k1, k2 = self.kind_of(n1), self.kind_of(n2)
                            kinds = {k1, k2}
                            if kinds == {"operation", "data"}:
                                process_op_data_pair(n1, n2)
                                progressed = True

                # 两节点环处理后，继续常规推进一轮
                if progressed:
                    head = len(q)
                    while head < len(q):
                        u = q[head]
                        head += 1
                        if u not in processed:
                            process_node(u)

            # 若仍然有未处理节点（更复杂的环），不给出错误，但给出提示，并按稳定顺序追加
            if len(processed) != len(self._nodes):
                # 将剩余节点按名称排序追加，确保返回全序列以便后续流程继续
                pending = sorted([nid for nid in self._nodes.keys() if nid not in processed])
                print(
                    "Warning: Unresolved cycles beyond two-node op-data pairs exist; appending remaining nodes in stable order: "
                    + ", ".join(pending)
                )
                order.extend(pending)

        return order

    def _detect_reachable(self, src: NodeId, target: NodeId) -> bool:
        stack = [src]
        seen = set()
        while stack:
            u = stack.pop()
            if u == target:
                return True
            if u in seen:
                continue
            seen.add(u)
            stack.extend(self._out_edges[u])
            stack.extend(self._ctrl_out_edges[u])
        return False

    def input_pairs(self, nid: NodeId) -> Dict[int, Data]:
        preds = self.predecessors(nid)
        ports = self.in_ports(nid)
        pairs = {0 if p is None else p: self.node(src) for src, p in zip(preds, ports)}
        return pairs
    
    def output_pairs(self, nid: NodeId) -> Dict[int, Data]:
        succs = self.successors(nid)
        ports = self.out_ports(nid)
        pairs = {0 if p is None else p: self.node(succ) for succ, p in zip(succs, ports)}
        return pairs
    
    def control_input_pairs(self, nid: NodeId) -> Dict[int, Any]:
        preds = self.control_predecessors(nid)
        ports = self.control_in_ports(nid)
        pairs = {0 if p is None else p: self.node(src) for src, p in zip(preds, ports)}
        return pairs

    def control_output_pairs(self, nid: NodeId) -> Dict[int, Any]:
        succs = self.control_successors(nid)
        ports = self.control_out_ports(nid)
        pairs = {0 if p is None else p: self.node(succ) for succ, p in zip(succs, ports)}
        return pairs

    def infer(self) -> None:
        order = self.topological_order()
        for nid in order:
            kind = self.kind_of(nid)
            node = self.node(nid)
            
            if kind == "view":
                # 处理View节点：需要找到其源Data节点并解析
                view_node: ViewData = node
                source_data_id = None
                
                # 查找连接到此View节点的Data节点
                assert len(self.predecessors(nid)) == 1, f"View node {nid} should have exactly one predecessor"
                for pred_id in self.predecessors(nid):
                    if self.kind_of(pred_id) in ("data","concat"):
                        source_data_id = pred_id
                        break
                
                self.node(nid).infer_shape_and_dtype(self.node(source_data_id))
                
            elif kind in ("operation",):
                pairs = self.input_pairs(nid)
                # pairs = sorted(pairs.items())
                infer = getattr(node, "infer", None)
                outs = infer(pairs) if callable(infer) else []
                succs = self.successors(nid)
                oports = self.out_ports(nid)
                out_pairs = [
                    (0 if p is None else p, succ)
                    for succ, p in zip(succs, oports)
                    if self.kind_of(succ) in ("data", "view")
                ]
                assert len(outs) == len(out_pairs), f"Op {node.name} inferred {len(outs)} outputs, but got {len(out_pairs)}"
                out_pairs.sort(key=lambda x: x[0])
                for idx, (p, succ) in enumerate(out_pairs):
                    assert idx == p, f"Output port mismatch for {node.name}: expected port {idx}, got {p}"
                    if idx >= len(outs):
                        break
                    shape, dtype = outs[idx]
                    d: Data = self.node(succ)
                    new_shape = d.shape if d.shape is not None else shape
                    if d.shape is not None and shape is not None and d.shape != shape:
                        raise ValueError(
                            f"Shape mismatch for {d.name}: have {d.shape}, infer {shape}"
                        )
                    new_dtype = d.dtype if d.dtype is not None else dtype
                    if d.dtype is not None and dtype is not None and d.dtype != dtype:
                        raise ValueError(
                            f"DType mismatch for {d.name}: have {d.dtype}, infer {dtype}"
                        )
                    self._nodes[succ] = Data(d.name, new_shape, new_dtype, d.tags)
            elif kind == "concat":
                concat_inputs = self.input_pairs(nid)
                node.validate(concat_inputs)

    def to_prim(self) -> None:
        existing_non_primitive = True
        while existing_non_primitive:
            order = self.topological_order()
            existing_non_primitive = False
            for nid in order:
                kind = self.kind_of(nid)
                node = self.node(nid)
                if kind in ("operation",) and not getattr(node, "primitive", True):
                    existing_non_primitive = True
                    to_prim = getattr(node, "to_prim", None)
                    if callable(to_prim):
                        result = to_prim()
                        if result is None:
                            continue
                        subgraph = result.get("subgraph", None)
                        input_mapping = result.get("input_mapping", {})
                        output_mapping = result.get("output_mapping", {})
                        if subgraph is not None:
                            self.replace_node_with_subgraph(nid, subgraph, input_mapping, output_mapping)
                    break
    
    def add_graph(self, other: "Graph") -> Dict[NodeId, NodeId]:
        idmap: Dict[NodeId, NodeId] = {}
        edge_usage: Dict[Tuple[NodeId, NodeId], int] = {}
        for oid, onode in other.nodes():
            nid = self.add_node(onode)
            idmap[oid] = nid
        for oid, _ in other.nodes():
            for idx, succ in enumerate(other._out_edges[oid]):
                sp = other._out_ports[oid][idx]
                dp = None
                key = (oid, succ)
                occurrence = edge_usage.get(key, 0)
                match_idx = -1
                count = 0
                for in_idx, src2 in enumerate(other._in_edges[succ]):
                    if src2 == oid:
                        if count == occurrence:
                            match_idx = in_idx
                            break
                        count += 1
                assert match_idx != -1
                dp = other._in_ports[succ][match_idx]
                edge_usage[key] = occurrence + 1
                self.connect(idmap[oid], idmap[succ], src_port=sp, dst_port=dp)
        for oid, _ in other.nodes():
            for idx, succ in enumerate(other._ctrl_out_edges[oid]):
                sp = other._ctrl_out_ports[oid][idx]
                dp = None
                for in_idx, src2 in enumerate(other._ctrl_in_edges[succ]):
                    if src2 == oid:
                        dp = other._ctrl_in_ports[succ][in_idx]
                        break
                self.connect_control(idmap[oid], idmap[succ], src_port=sp, dst_port=dp)
        return idmap

    def replace_node_with_subgraph(
        self, 
        node_id: NodeId, 
        subgraph: "Graph", 
        input_mapping: Dict[int, Tuple[NodeId, int]], 
        output_mapping: Dict[int, Tuple[NodeId, int]]
    ) -> Dict[NodeId, NodeId]:
        """将指定节点替换为更复杂的子图
        
        Args:
            node_id: 要被替换的节点ID
            subgraph: 用于替换的子图
            input_mapping: 输入端口映射，格式为 {原节点输入端口: (子图节点ID, 子图节点端口)}
            output_mapping: 输出端口映射，格式为 {原节点输出端口: (子图节点ID, 子图节点端口)}
            
        Returns:
            Dict[NodeId, NodeId]: 子图节点ID到当前图中新节点ID的映射
            
        Raises:
            ValueError: 当指定的节点不存在或映射参数无效时
        """
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} does not exist in the graph")
        
        # 收集原节点的所有连接信息
        input_connections = []  # [(pred_node, pred_src_port, dst_port)]
        output_connections = []  # [(succ_node, src_port, succ_dst_port)]
        ctrl_input_connections = []  # [pred_node]
        ctrl_output_connections = []  # [succ_node]
        
        # 收集数据输入连接
        for i, pred in enumerate(self._in_edges[node_id]):
            dst_port = self._in_ports[node_id][i]
            # 找到前驱节点中对应的源端口
            pred_src_port = None
            for j, pred_succ in enumerate(self._out_edges[pred]):
                if pred_succ == node_id:
                    pred_src_port = self._out_ports[pred][j]
                    break
            input_connections.append((pred, pred_src_port, dst_port))
        
        # 收集数据输出连接
        for i, succ in enumerate(self._out_edges[node_id]):
            src_port = self._out_ports[node_id][i]
            # 找到后继节点中对应的目标端口
            succ_dst_port = None
            for j, succ_pred in enumerate(self._in_edges[succ]):
                if succ_pred == node_id:
                    succ_dst_port = self._in_ports[succ][j]
                    break
            output_connections.append((succ, src_port, succ_dst_port))
        
        # 收集控制输入连接
        ctrl_input_connections = self._ctrl_in_edges[node_id].copy()
        
        # 收集控制输出连接
        ctrl_output_connections = self._ctrl_out_edges[node_id].copy()
        
        # 断开原节点的所有连接
        self._disconnect_node(node_id)
        
        # 移除原节点
        del self._nodes[node_id]
        del self._in_edges[node_id]
        del self._out_edges[node_id]
        del self._in_ports[node_id]
        del self._out_ports[node_id]
        del self._ctrl_in_edges[node_id]
        del self._ctrl_out_edges[node_id]
        
        # 添加子图到当前图中
        id_mapping = self.add_graph(subgraph)
        
        def _normalize_mapping_entries(value: Optional[object]) -> List[Tuple[NodeId, Optional[int]]]:
            if value is None:
                return []
            # 单个 (node, port) 元组
            if (
                isinstance(value, tuple)
                and len(value) == 2
                and not isinstance(value[0], (list, tuple))
            ):
                return [(value[0], value[1])]
            entries: List[Tuple[NodeId, Optional[int]]] = []
            if isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, tuple) and len(item) == 2:
                        entries.append((item[0], item[1]))
                    elif isinstance(item, list) and len(item) == 2:
                        entries.append((item[0], item[1]))
                    else:
                        raise ValueError(
                            "Mapping entries must be sequences of length 2, got invalid value"
                        )
                return entries
            raise ValueError("Mapping value must be a tuple or a list/tuple of tuples")

        def _iter_mapping_nodes(mapping: Dict[int, object]) -> Iterable[Tuple[NodeId, Optional[int]]]:
            for value in mapping.values():
                for node_id, port in _normalize_mapping_entries(value):
                    yield node_id, port

        # 重新连接输入边
        for pred, pred_src_port, dst_port in input_connections:
            if dst_port in input_mapping:
                input_mapping_entries = _normalize_mapping_entries(input_mapping[dst_port])
                for target_node_id, target_port in input_mapping_entries:
                    if target_node_id not in id_mapping:
                        raise ValueError(f"Target node {target_node_id} not found in subgraph")

                    mapped_target = id_mapping[target_node_id]
                    self.connect(pred, mapped_target, src_port=pred_src_port, dst_port=target_port)
        
        # 重新连接输出边
        for succ, src_port, succ_dst_port in output_connections:
            if src_port in output_mapping:
                output_mapping_entries = _normalize_mapping_entries(output_mapping[src_port])
                for source_node_id, source_port in output_mapping_entries:
                    if source_node_id not in id_mapping:
                        raise ValueError(f"Source node {source_node_id} not found in subgraph")

                    mapped_source = id_mapping[source_node_id]
                    self.connect(mapped_source, succ, src_port=source_port, dst_port=succ_dst_port)
        
        # 重新连接控制边：原控制输入指向子图入口节点，子图出口节点指向原控制输出
        # subgraph_node_ids = set(id_mapping.values())

        def _is_non_data_node(nid: NodeId) -> bool:
            return self.kind_of(nid) not in ("data", "view", "concat")

        mapped_entry_nodes = {
            id_mapping[target_node_id]
            for target_node_id, _ in _iter_mapping_nodes(input_mapping)
            if target_node_id in id_mapping and _is_non_data_node(id_mapping[target_node_id])
        }
        subgraph_entry_nodes: Set[NodeId] = set(mapped_entry_nodes)
        if not subgraph_entry_nodes and ctrl_input_connections:
            raise ValueError(
                f"Cannot reconnect control inputs for node '{node_id}': no entry nodes in subgraph mapping."
            )

        for pred in ctrl_input_connections:
            for entry_node in subgraph_entry_nodes:
                if entry_node not in self._ctrl_out_edges.get(pred, []):
                    self.connect_control(pred, entry_node)
        
        mapped_exit_nodes = {
            id_mapping[source_node_id]
            for source_node_id, _ in _iter_mapping_nodes(output_mapping)
            if source_node_id in id_mapping and _is_non_data_node(id_mapping[source_node_id])
        }
        subgraph_exit_nodes: Set[NodeId] = set(mapped_exit_nodes)
        if not subgraph_exit_nodes and ctrl_output_connections:
            raise ValueError(
                f"Cannot reconnect control outputs for node '{node_id}': no exit nodes in subgraph mapping."
            )
        
        for succ in ctrl_output_connections:
            for exit_node in subgraph_exit_nodes:
                if succ not in self._ctrl_out_edges[exit_node]:
                    self.connect_control(exit_node, succ)
        
        return id_mapping
    
    def _disconnect_node(self, node_id: NodeId) -> None:
        """断开指定节点的所有连接（内部辅助方法）"""
        # 断开输入数据边
        for pred in self._in_edges[node_id]:
            # 从前驱节点的输出列表中移除当前节点
            while node_id in self._out_edges[pred]:
                idx = self._out_edges[pred].index(node_id)
                self._out_edges[pred].pop(idx)
                self._out_ports[pred].pop(idx)
        
        # 断开输出数据边
        for succ in self._out_edges[node_id]:
            # 从后继节点的输入列表中移除当前节点
            while node_id in self._in_edges[succ]:
                idx = self._in_edges[succ].index(node_id)
                self._in_edges[succ].pop(idx)
                self._in_ports[succ].pop(idx)
        
        # 断开控制输入边
        for pred in self._ctrl_in_edges[node_id]:
            if node_id in self._ctrl_out_edges[pred]:
                self._ctrl_out_edges[pred].remove(node_id)
        
        # 断开控制输出边
        for succ in self._ctrl_out_edges[node_id]:
            if node_id in self._ctrl_in_edges[succ]:
                self._ctrl_in_edges[succ].remove(node_id)

    def disconnect(self, src: NodeId, dst: NodeId) -> None:
        """断开指定的有向边（数据边或控制边）"""
        if dst in self._out_edges[src]:
            # 断开数据边
            idx = self._out_edges[src].index(dst)
            self._out_edges[src].pop(idx)
            self._out_ports[src].pop(idx)
            idx = self._in_edges[dst].index(src)
            self._in_edges[dst].pop(idx)
            self._in_ports[dst].pop(idx)
        elif dst in self._ctrl_out_edges[src]:
            # 断开控制边
            self._ctrl_out_edges[src].remove(dst)
            self._ctrl_in_edges[dst].remove(src)
        else:
            raise ValueError(f"No edge exists from {src} to {dst}")
    
    def to_dot(self, vertical: bool = True) -> str:
        """生成Graphviz DOT格式的可视化描述
        
        DOT格式特点：
        - 使用有向图(digraph)表示DAG结构
        - 不同类型的节点使用不同的形状和颜色
        - 数据边显示端口信息，控制边使用虚线表示
        
        节点样式：
        - 数据节点：椭圆形，显示名称、形状、数据类型
        - 操作节点：矩形，显示名称
        - 控制节点：菱形，灰色填充，显示名称
        
        边样式：
        - 数据边：实线，标注端口信息（s<端口>表示源端口，d<端口>表示目标端口）
        - 控制边：虚线，灰色
        
        Args:
            vertical: 是否使用纵向布局（True为上下布局，False为左右布局）
        
        Returns:
            str: DOT格式的图描述字符串，可用于Graphviz渲染
        """
        lines = ["digraph minigraph {"]
        # 设置布局方向：TB为上下（纵向），LR为左右（横向）
        rankdir = "TB" if vertical else "LR"
        lines.append(f"  rankdir={rankdir};")
        lines.append("  node [fontsize=10];")  # 设置默认字体大小
        lines.append("  edge [fontsize=8];")   # 设置边标签字体大小
        
        # 生成节点定义
        for nid, node in self._nodes.items():
            node_type = self.kind_of(nid)
            # 处理节点ID中的特殊字符，确保DOT格式兼容性
            safe_nid = nid.replace(":", "_").replace("-", "_")
            
            if node_type == "data":
                # 数据节点：椭圆形，显示完整信息
                d: Data = node
                # 构建显示标签，处理None值
                shape_str = str(d.shape) if d.shape else "Unknown"
                dtype_str = d.dtype.value if d.dtype else "Unknown"
                label = f"{d.name}\\nShape: {shape_str}\\nType: {dtype_str}"
                
                # 检查是否有payload（包括memref中的payload）
                has_payload = (d.payload is not None) or (d.memref is not None and d.memref.payload is not None)
                
                # 根据是否有payload设置不同的边框样式
                if has_payload:
                    # 有payload：使用粗边框（penwidth=3）
                    style_attr = 'style=filled, fillcolor=lightblue, penwidth=3, color=darkblue'
                else:
                    # 没有payload：使用普通边框
                    style_attr = 'style=filled, fillcolor=lightblue'
                
                lines.append(
                    f'  "{safe_nid}" [label="{label}", shape=ellipse, '
                    f'{style_attr}];'
                )
            
            elif node_type == "view":
                # View节点：六边形，显示视图信息
                v: ViewData = node
                # 根据View类型构建标签
                if hasattr(v, 'shape') and v.shape:
                    shape_str = str(v.shape)
                elif hasattr(v, 'target_shape') and v.target_shape:
                    shape_str = str(v.target_shape)
                else:
                    shape_str = "Inferred"
                
                dtype_str = v.dtype.value if v.dtype else "Inherited"
                
                if v.view_type == "reshape":
                    view_info = f"Reshape to {v.target_shape}"
                elif v.view_type == "slice":
                    view_info = f"Slice [{v.slice_start}:{v.slice_end}]"
                else:
                    view_info = f"View: {v.view_type}"
                
                label = f"{v.name}\\n{view_info}\\nShape: {shape_str}\\nType: {dtype_str}"
                
                lines.append(
                    f'  "{safe_nid}" [label="{label}", shape=hexagon, '
                    f'style=filled, fillcolor=lightgreen];'
                )
                
            elif node_type == "concat":
                # ConcatData 节点：双八边形，强调拼接语义
                c: ConcatData = node
                shape_str = str(c.shape) if getattr(c, 'out_shape', None) else "Unknown"
                dtype_str = c.dtype.value if getattr(c, 'dtype', None) else "Unknown"
                label = f"{c.name}\\nConcat\\nOutShape: {shape_str}\\nType: {dtype_str}"
                lines.append(
                    f'  "{safe_nid}" [label="{label}", shape=doubleoctagon, '
                    f'style=filled, fillcolor=lightsalmon];'
                )
                
            elif node_type == "control":
                # 控制节点：菱形，灰色填充
                ctrl_name = getattr(node, "name", f"ctrl_{nid}")
                lines.append(
                    f'  "{safe_nid}" [label="{ctrl_name}", shape=diamond, '
                    f'style=filled, fillcolor=lightgrey];'
                )
                
            else:  # operation节点
                # 操作节点：矩形，默认颜色
                op_name = getattr(node, "name", f"op_{nid}")
                lines.append(
                    f'  "{safe_nid}" [label="{op_name}", shape=box, '
                    f'style=filled, fillcolor=lightyellow];'
                )
        
        # 生成数据边（带端口标签）
        for src_nid in self._nodes.keys():
            safe_src = src_nid.replace(":", "_").replace("-", "_")
            
            for idx, dst_nid in enumerate(self._out_edges[src_nid]):
                safe_dst = dst_nid.replace(":", "_").replace("-", "_")
                
                # 获取端口信息
                src_port = self._out_ports[src_nid][idx]
                
                # 查找对应的目标端口
                dst_port = None
                for in_idx, in_src in enumerate(self._in_edges[dst_nid]):
                    if in_src == src_nid:
                        dst_port = self._in_ports[dst_nid][in_idx]
                        break
                
                # 构建端口标签
                port_labels = []
                if src_port is not None:
                    port_labels.append(f"s{src_port}")
                if dst_port is not None:
                    port_labels.append(f"d{dst_port}")
                
                # 生成边定义
                if port_labels:
                    label_str = f' [label="{"/".join(port_labels)}", color=blue]'
                else:
                    label_str = ' [color=blue]'
                
                lines.append(f'  "{safe_src}" -> "{safe_dst}"{label_str};')
        
        # 生成控制边（虚线样式）
        for src_nid in self._nodes.keys():
            safe_src = src_nid.replace(":", "_").replace("-", "_")
            
            for dst_nid in self._ctrl_out_edges[src_nid]:
                safe_dst = dst_nid.replace(":", "_").replace("-", "_")
                lines.append(
                    f'  "{safe_src}" -> "{safe_dst}" '
                    f'[style=dashed, color=gray, label="ctrl"];'
                )
        
        lines.append("}")
        return "\n".join(lines)

    def visualize(self, output_path: str = "graph", format: str = "png", vertical: bool = True) -> str:
        """生成并保存图的可视化图片
        
        使用Graphviz渲染DOT格式描述，生成图片文件。
        需要系统安装graphviz工具。
        
        Args:
            output_path: 输出文件路径（不包含扩展名）
            format: 输出格式，支持 "png", "pdf", "svg", "jpg" 等
            vertical: 是否使用纵向布局
            
        Returns:
            str: 生成的图片文件完整路径
            
        Raises:
            ImportError: 当graphviz包未安装时
            Exception: 当图片生成失败时
        """
        try:
            import graphviz
        except ImportError:
            raise ImportError(
                "需要安装graphviz包来生成图片。请运行: pip install graphviz\n"
                "同时确保系统已安装Graphviz软件: https://graphviz.org/download/"
            )
        
        # 生成DOT格式描述
        dot_content = self.to_dot(vertical=vertical)
        
        try:
            # 创建Graphviz源对象
            source = graphviz.Source(dot_content)
            
            # 渲染并保存图片
            output_file = source.render(
                filename=output_path,
                format=format,
                cleanup=True  # 删除临时.dot文件
            )
            
            print(f"图形已保存到: {output_file}")
            return output_file
            
        except Exception as e:
            raise Exception(f"生成图片时出错: {str(e)}")

    def save_dot(self, output_path: str = "graph.dot", vertical: bool = True) -> str:
        """保存DOT格式文件
        
        Args:
            output_path: 输出文件路径
            vertical: 是否使用纵向布局
            
        Returns:
            str: 保存的文件路径
        """
        dot_content = self.to_dot(vertical=vertical)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(dot_content)
        
        print(f"DOT文件已保存到: {output_path}")
        return output_path
    
    def extract_subgraph(
        self, 
        nodes: Optional[List[NodeId]] = None,
        start_nodes: Optional[List[NodeId]] = None,
        end_nodes: Optional[List[NodeId]] = None,
    ) -> "Graph":
        """提取子图并创建新的Graph实例
        
        提供多种方式定义子图：
        1. 直接指定节点列表
        2. 指定起始和结束节点，自动推导中间路径
        
        Args:
            nodes: 直接指定要包含在子图中的节点ID列表
            start_nodes: 起始节点列表，从这些节点开始向前搜索
            end_nodes: 结束节点列表，搜索到这些节点结束
            
        Returns:
            Graph: 包含指定节点的新图实例
            
        Raises:
            ValueError: 当输入参数无效或节点不存在时
            
        Examples:
            # 方式1: 直接指定节点
            subgraph = graph.extract_subgraph(nodes=["conv1", "relu1", "data1"])
            
            # 方式2: 指定起始和结束节点
            subgraph = graph.extract_subgraph(start_nodes=["input"], end_nodes=["output"])
            
            # 方式3: 只指定起始节点，包含所有后续依赖
            subgraph = graph.extract_subgraph(start_nodes=["conv1"])
        """
        # 参数验证
        if not nodes and not start_nodes and not end_nodes:
            raise ValueError("必须至少指定一种节点选择方式：nodes、start_nodes或end_nodes")
        
        if nodes and (start_nodes or end_nodes):
            raise ValueError("不能同时指定nodes参数和start_nodes/end_nodes参数")
        
        # 确定目标节点集合
        target_nodes = set()
        
        if nodes:
            # 方式1: 直接指定节点列表
            for node_id in nodes:
                if node_id not in self._nodes:
                    raise ValueError(f"节点 {node_id} 不存在于图中")
                target_nodes.add(node_id)
                
        else:
            # 方式2: 通过起始和结束节点推导
            if start_nodes:
                for node_id in start_nodes:
                    if node_id not in self._nodes:
                        raise ValueError(f"起始节点 {node_id} 不存在于图中")
            
            if end_nodes:
                for node_id in end_nodes:
                    if node_id not in self._nodes:
                        raise ValueError(f"结束节点 {node_id} 不存在于图中")
            
            # 使用路径查找算法确定目标节点
            target_nodes = self.find_path_nodes(start_nodes or [], end_nodes or [])
        
        # 创建子图
        return self._create_subgraph_from_nodes(target_nodes)
    
    def find_path_nodes(self, start_nodes: List[NodeId], end_nodes: List[NodeId]) -> set:
        """查找从起始节点到结束节点的所有路径上的节点"""
        target_nodes = set()
        
        # 如果只有起始节点，包含所有可达节点
        if start_nodes and not end_nodes:
            for start_node in start_nodes:
                target_nodes.update(self._get_reachable_nodes(start_node))
        
        # 如果只有结束节点，包含所有能到达结束节点的节点
        elif end_nodes and not start_nodes:
            for end_node in end_nodes:
                target_nodes.update(self._get_predecessor_nodes(end_node))
        
        # 如果同时有起始和结束节点，查找路径
        elif start_nodes and end_nodes:
            for start_node in start_nodes:
                for end_node in end_nodes:
                    path_nodes = self._find_path_between_nodes(start_node, end_node)
                    target_nodes.update(path_nodes)
        
        return target_nodes
    
    def _get_reachable_nodes(self, start_node: NodeId) -> set:
        """获取从指定节点可达的所有节点（包括自身）"""
        visited = set()
        stack = [start_node]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            
            visited.add(current)
            
            # 添加数据流后继节点
            stack.extend(self._out_edges[current])
            
            # 添加控制流后继节点
            stack.extend(self._ctrl_out_edges[current])
        
        return visited
    
    def _get_predecessor_nodes(self, end_node: NodeId) -> set:
        """获取能到达指定节点的所有前驱节点（包括自身）"""
        visited = set()
        stack = [end_node]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            
            visited.add(current)
            
            # 添加数据流前驱节点
            stack.extend(self._in_edges[current])
            
            # 添加控制流前驱节点
            stack.extend(self._ctrl_in_edges[current])
        
        return visited
    
    def _find_path_between_nodes(self, start_node: NodeId, end_node: NodeId) -> set:
        """查找两个节点之间的所有路径上的节点"""
        # 从起始节点开始的可达节点
        reachable_from_start = self._get_reachable_nodes(start_node)
        
        # 能到达结束节点的前驱节点
        predecessors_of_end = self._get_predecessor_nodes(end_node)
        
        # 交集就是路径上的节点
        return reachable_from_start.intersection(predecessors_of_end)
    
    def _create_subgraph_from_nodes(self, target_nodes: set) -> "Graph":
        """从节点集合创建子图"""
        subgraph = Graph()
        node_mapping = {}
        
        # 1. 添加节点到子图
        for node_id in target_nodes:
            original_node = self._nodes[node_id]
            # 创建节点的深拷贝以避免修改原图
            from copy import deepcopy
            new_node = deepcopy(original_node)
            
            # 添加到子图并记录映射关系
            new_node_id = subgraph.add_node(new_node)
            node_mapping[node_id] = new_node_id
        
        # 2. 添加数据边到子图
        for src_id in target_nodes:
            src_mapped = node_mapping[src_id]
            
            for idx, dst_id in enumerate(self._out_edges[src_id]):
                if dst_id in target_nodes:  # 只有当目标节点也在子图中时才添加边
                    dst_mapped = node_mapping[dst_id]
                    src_port = self._out_ports[src_id][idx]
                    
                    # 查找对应的目标端口
                    dst_port = None
                    for in_idx, in_src in enumerate(self._in_edges[dst_id]):
                        if in_src == src_id:
                            dst_port = self._in_ports[dst_id][in_idx]
                            break
                    
                    # 使用内部方法直接连接，避免重复验证
                    subgraph._connect_nocheck(
                        src_mapped, dst_mapped,
                        src_port=src_port, dst_port=dst_port
                    )
        
        # 3. 添加控制边到子图
        for src_id in target_nodes:
            src_mapped = node_mapping[src_id]
            
            for dst_id in self._ctrl_out_edges[src_id]:
                if dst_id in target_nodes:  # 只有当目标节点也在子图中时才添加边
                    dst_mapped = node_mapping[dst_id]
                    subgraph.connect_control(src_mapped, dst_mapped)
        
        return subgraph
    
    # 分割图
    def split(self, split_edges: List[Tuple[NodeId, NodeId]]) -> List[List[NodeId]]:
        """分割图为多个连通子图
        
        Args:
            split_edges: 需要断开的边列表，每个元素为(src_node, dst_node)元组
            
        Returns:
            List[List[NodeId]]: 分割后各个连通分量的节点列表
        """
        split_set = set(split_edges)
        visited = set()
        components = []
        
        def dfs(node: NodeId, component: List[NodeId]) -> None:
            if node in visited:
                return
            visited.add(node)
            component.append(node)
            
            # 遍历数据流连接（忽略分割边）
            for succ in self._out_edges[node]:
                if (node, succ) not in split_set:
                    dfs(succ, component)
            for pred in self._in_edges[node]:
                if (pred, node) not in split_set:
                    dfs(pred, component)
                    
            # 遍历控制流连接（忽略分割边）
            for succ in self._ctrl_out_edges[node]:
                if (node, succ) not in split_set:
                    dfs(succ, component)
            for pred in self._ctrl_in_edges[node]:
                if (pred, node) not in split_set:
                    dfs(pred, component)
        
        for node_id in self._nodes:
            if node_id not in visited:
                component = []
                dfs(node_id, component)
                components.append(component)
                
        return components
        