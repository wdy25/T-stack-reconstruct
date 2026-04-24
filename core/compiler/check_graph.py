from typing import Dict, List, Tuple, Optional, Any

from core.ir.graph import Graph, NodeId
from core.ir.hardware_graph import HardwareGraph
from core.ir.data import Data, ViewData, MemBlock
from core.ir.operation import Operation
from core.ir.communication_op import CommOp
from core.ir.control_op import ControlOp


class CheckGraph:
	"""检查 HardwareGraph 是否满足生成配置所需的前置条件。

	检查项（按用户要求）：
	1. 所有节点都指定了 core_id
	2. 所有数据节点（Data）都有地址（memref.addr 非负）
	3. 所有没有输入的数据节点（入度为 0 的 Data）都有 payload（并且其 memref 也有 payload）
	4. 所有 operation、communication、control 节点都同时具备 gen_prim 函数和 para_node 函数
	5. 没有完全不与其他节点连接（数据边与控制边都没有）的孤立节点

	用法：
		checker = CheckGraph()
		ok = checker.check(hg)
	"""

	def check(self, hg: HardwareGraph) -> bool:
		issues = {
			"missing_core_id": self._check_core_ids(hg),
			"data_without_addr": self._check_data_addresses(hg),
			"source_data_without_payload": self._check_source_data_payload(hg),
			"nodes_missing_required_methods": self._check_required_methods(hg),
			"isolated_nodes": self._check_isolated_nodes(hg),
		}

		self._print_report(hg, issues)
		# 如果所有列表都为空，则为通过
		return all(len(v) == 0 for v in issues.values())

	# ========== 具体检查 ==========
	def _check_core_ids(self, hg: HardwareGraph) -> List[NodeId]:
		"""检查所有节点是否有 core_id。接受 int 或 2 元组，只要非 None 即视为指定。"""
		missing: List[NodeId] = []

		# 访问内部映射以避免 get_core_id 的类型不一致导致异常
		core_map: Dict[NodeId, Any] = getattr(hg, "_core_id_dict", {})

		for nid, _ in hg.nodes():
			if nid not in core_map:
				missing.append(nid)
				continue
			cid = core_map.get(nid, None)
			if cid is None:
				missing.append(nid)
				continue
			# 允许 int 或 tuple 形式；若为 tuple，要求长度>0 且元素为非负整数；若为 int 要求非负
			if isinstance(cid, tuple):
				if len(cid) == 0 or any((not isinstance(x, int)) or (x < 0) for x in cid):
					missing.append(nid)
			elif isinstance(cid, int):
				if cid < 0:
					missing.append(nid)
			else:
				# 其他类型视为未正确指定
				missing.append(nid)

		return missing

	def _check_data_addresses(self, hg: HardwareGraph) -> List[NodeId]:
		"""检查所有 Data 节点的 memref.addr 是否有效（>=0）。"""
		bad: List[NodeId] = []
		for nid, node in hg.nodes():
			if hg.kind_of(nid) == "data":
				d: Data = node  # type: ignore[assignment]
				if d.memref is None or d.memref.addr is None or d.memref.addr < 0:
					bad.append(nid)
		return bad

	def _check_source_data_payload(self, hg: HardwareGraph) -> List[NodeId]:
		"""检查入度为 0 的 Data 节点：要求 data.memref.payload 都不为 None。"""
		bad: List[NodeId] = []
		for nid, node in hg.nodes():
			if hg.kind_of(nid) == "data":
				if len(hg.predecessors(nid)) == 0:
					d: Data = node  # type: ignore[assignment]
					mem_ok = d.memref is not None and getattr(d.memref, "payload", None) is not None
					if not mem_ok:
						bad.append(nid)
		return bad

	def _check_required_methods(self, hg: HardwareGraph) -> List[NodeId]:
		"""检查 operation/communication/control 节点是否同时具有 gen_prim 与 para_node 方法。"""
		missing_methods: List[NodeId] = []
		for nid, node in hg.nodes():
			kind = hg.kind_of(nid)
			if kind in ("operation", "communication", "control"):
				has_gen_prim = hasattr(node, "gen_prim") and callable(getattr(node, "gen_prim", None))
				has_para_node = hasattr(node, "para_node") and callable(getattr(node, "para_node", None))
				if not (has_gen_prim and has_para_node):
					missing_methods.append(nid)
		return missing_methods

	def _check_isolated_nodes(self, hg: HardwareGraph) -> List[NodeId]:
		"""检查没有任何数据边或控制边连接的孤立节点。"""
		isolated: List[NodeId] = []
		for nid, _ in hg.nodes():
			if (
				len(hg.predecessors(nid)) == 0
				and len(hg.successors(nid)) == 0
				and len(hg.control_predecessors(nid)) == 0
				and len(hg.control_successors(nid)) == 0
			):
				isolated.append(nid)
		return isolated

	# ========== 输出报告 ==========
	def _print_report(self, hg: HardwareGraph, issues: Dict[str, List[NodeId]]) -> None:
		def fmt_nodes(nodes: List[NodeId]) -> str:
			if not nodes:
				return "  - 无"
			return "\n".join([f"  - {nid} ({hg.kind_of(nid)})" for nid in nodes])

		print("\n===== HardwareGraph 检查报告 =====")

		print("\n[1] 未指定或非法的 core_id：")
		print(fmt_nodes(issues["missing_core_id"]))

		print("\n[2] 缺少内存地址的 Data 节点（memref.addr < 0 或未设置）：")
		print(fmt_nodes(issues["data_without_addr"]))

		print("\n[3] 入度为 0 但缺少 payload 的 Data 节点（data.payload 以及 memref.payload 均需存在）：")
		print(fmt_nodes(issues["source_data_without_payload"]))

		print("\n[4] 缺少 gen_prim 或 para_node 的节点（operation / communication / control）：")
		print(fmt_nodes(issues["nodes_missing_required_methods"]))

		print("\n[5] 完全孤立（无任意连接）的节点：")
		print(fmt_nodes(issues["isolated_nodes"]))

		all_ok = all(len(v) == 0 for v in issues.values())
		print("\n总体结果：{}".format("PASS ✅" if all_ok else "FAIL ❌"))

