import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple
from core.ir.graph import Graph, NodeId
from core.ir.data import Data, ViewData, data2Mem, MemBlock, resolve_view_data, ConcatData, resolve_concat_data
from core.ir.operation import Operation
from core.ir.control_op import ControlOp
from core.ir.communication_op import CommOp, SendOp, RecvOp


class HardwareGraph(Graph):
    """Hardware-aware graph that extends the base Graph with core mapping and hardware-specific features.
    
    This class adds hardware-specific functionality to the base graph:
    1. Core ID mapping for nodes (to support multi-core deployment)
    2. Automatic parameter node generation for operations
    3. Memory reference generation for data nodes
    
    Attributes:
        _core_id_dict (Dict[NodeId, int]): Maps each node to its assigned core ID.
    """

    def __init__(self, original_graph: Graph = None) -> None:
        super().__init__()
        self._core_id_dict: Dict[NodeId, Tuple[int, int]] = {}
        self.tag_id = 0
        if original_graph:
            # Deep copy nodes and edges from the original graph
            self._nodes = {nid: node for nid, node in original_graph._nodes.items()}
            self._in_edges = {nid: edges[:] for nid, edges in original_graph._in_edges.items()}
            self._out_edges = {nid: edges[:] for nid, edges in original_graph._out_edges.items()}
            self._ctrl_in_edges = {nid: edges[:] for nid, edges in original_graph._ctrl_in_edges.items()}
            self._ctrl_out_edges = {nid: edges[:] for nid, edges in original_graph._ctrl_out_edges.items()}
            self._in_ports = {nid: ports[:] for nid, ports in original_graph._in_ports.items()}
            self._out_ports = {nid: ports[:] for nid, ports in original_graph._out_ports.items()}
            self._ctrl_in_ports = {nid: ports[:] for nid, ports in original_graph._ctrl_in_ports.items()}
            self._ctrl_out_ports = {nid: ports[:] for nid, ports in original_graph._ctrl_out_ports.items()}
            # Initialize core IDs to (0, 0) by default
            for nid in self._nodes.keys():
                self._core_id_dict[nid] = (0, 0)

    def add_node(self, node: Any, core_id: Optional[Tuple[int, int]] = None) -> NodeId:
        """Add a node to the graph with optional core ID assignment.
        
        Args:
            node: The node to add (Data, Operation, or ControlOp).
            core_id: Optional core ID to assign to this node. If None, defaults to (0, 0).
            
        Returns:
            NodeId: The ID of the added node.
        """
        nid = super().add_node(node)
        # Assign core ID (default to 0 if not specified)
        self._core_id_dict[nid] = core_id if core_id is not None else (0, 0)
        return nid

    def set_core_id(self, node_id: NodeId, core_id: Tuple[int, int]) -> None:
        """Set the core ID for a specific node.
        
        Args:
            node_id: The ID of the node to update.
            core_id: The core ID to assign.
            
        Raises:
            ValueError: If the node doesn't exist.
        """
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} does not exist in the graph")
        self._core_id_dict[node_id] = core_id

    def set_core_id_for_nodes(self, node_ids: Iterable[NodeId], core_id: Tuple[int, int]) -> None:
        """Set the core ID for multiple nodes.
        
        Args:
            node_ids: Iterable of node IDs to update.
            core_id: The core ID to assign to all specified nodes.
            
        Raises:
            ValueError: If any node doesn't exist.
        """
        for node_id in node_ids:
            if node_id not in self._nodes:
                raise ValueError(f"Node {node_id} does not exist in the graph")
            self._core_id_dict[node_id] = core_id
    
    def get_core_id(self, node_id: NodeId) -> Tuple[int, int]:
        """Get the core ID for a specific node.
        
        Args:
            node_id: The ID of the node to query.
            
        Returns:
            int: The core ID of the node.
            
        Raises:
            ValueError: If the node doesn't exist.
        """
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} does not exist in the graph")
        return self._core_id_dict[node_id]

    def get_nodes_by_core(self, core_id: Tuple[int, int]) -> List[NodeId]:
        """Get all nodes assigned to a specific core.
        
        Args:
            core_id: The core ID to query.
            
        Returns:
            List[NodeId]: List of node IDs assigned to the specified core.
        """
        return [nid for nid, cid in self._core_id_dict.items() if cid == core_id]

    def get_all_core_ids(self) -> List[Tuple[int, int]]:
        """Get all unique core IDs used in the graph.
        
        Returns:
            List[Tuple[int, int]]: Sorted list of unique core IDs.
        """
        return sorted(list(set(self._core_id_dict.values())))

    def gen_para_nodes(self) -> Dict[NodeId, NodeId]:
        """Generate parameter nodes for all operations in the graph.
        
        This function automatically traverses the graph, identifies all Operation nodes,
        and generates corresponding parameter nodes. Each parameter node is connected
        to its operation at the last available input port.
        
        Returns:
            Dict[NodeId, NodeId]: Mapping from operation node ID to its parameter node ID.
            
        Raises:
            ValueError: If an operation doesn't support parameter generation.
        """
        para_node_mapping = {}
        para_node_connection = {}
        
        # Iterate through all nodes to find operations
        for nid, node in self._nodes.items():
            node_type = self.kind_of(nid)
            
            if node_type in ["operation", "communication"]:
                operation = node
                
                # Check if the operation has a para_node method
                if hasattr(operation, 'para_node') and callable(getattr(operation, 'para_node')):
                    # Get input and output data for the operation
                    input_pairs = self.input_pairs(nid)
                    output_pairs = self.output_pairs(nid)
                    
                    # Generate parameter node
                    para_data = operation.para_node(input_pairs, output_pairs)
                    
                    if para_data is not None:                            
                        para_node_mapping[nid] = para_data
                        para_node_connection[nid] = operation.para_connection() if hasattr(operation, 'para_connection') and callable(getattr(operation, 'para_connection')) else False
                    else:
                        # warnings.warn(f"Operation '{operation.name}' generates None para node.")
                        pass
            
            elif node_type == "control":
                control_op = node
                
                # Check if the control operation has a para_node method
                if hasattr(control_op, 'para_node') and callable(getattr(control_op, 'para_node')):
                    # Get input and output data for the control operation
                    input_pairs = self.input_pairs(nid)
                    output_pairs = self.output_pairs(nid)
                    control_input_pairs = self.control_input_pairs(nid)
                    control_output_pairs = self.control_output_pairs(nid)
                    
                    # Generate parameter node
                    para_data = control_op.para_node(input_pairs, output_pairs)
                    
                    if para_data is not None:                            
                        para_node_mapping[nid] = para_data
                        para_node_connection[nid] = control_op.para_connection() if hasattr(control_op, 'para_connection') and callable(getattr(control_op, 'para_connection')) else False
                    else:
                        # warnings.warn(f"Control operation '{control_op.name}' generates None para node.")
                        pass
        
        for op_nid, para_data in para_node_mapping.items():
            # Add parameter node to the graph
            para_nid = self.add_node(para_data, core_id=self.get_core_id(op_nid))
            
            # Connect parameter node to the operation
            if para_node_connection[op_nid]:
                # Double connection (input and output)
                self.connect(para_nid, op_nid)
                self.connect(op_nid, para_nid)
            else:
                # Single connection (only input)
                self.connect(para_nid, op_nid)

        return para_node_mapping

    def gen_memref_for_all_data(self) -> Dict[NodeId, MemBlock]:
        """Generate memory references for all Data nodes in the graph.
        
        This function traverses all nodes in the graph, identifies Data nodes,
        and generates corresponding MemBlock references using the data2Mem function.
        The generated memref is stored in the Data node's memref field.
        
        Returns:
            Dict[NodeId, MemBlock]: Mapping from data node ID to its MemBlock reference.
            
        Raises:
            ValueError: If a Data node cannot be converted to MemBlock.
        """
        memref_mapping = {}
        
        # Iterate through all nodes to find data nodes
        for nid, node in self._nodes.items():
            node_type = self.kind_of(nid)
            
            if node_type == "data":
                data_node: Data = node
                if data_node.memref is not None:
                    # Skip if memref already exists
                    memref_mapping[nid] = data_node.memref
                    warnings.warn(f"Data node '{data_node.name}' already has a memref, skipping regeneration")
                    continue
                
                # Check if the data node has the required attributes for memref generation
                if data_node.shape is None:
                    print(f"Warning: Data node '{data_node.name}' has no shape, skipping memref generation")
                    continue
                    
                if data_node.dtype is None:
                    print(f"Warning: Data node '{data_node.name}' has no dtype, skipping memref generation")
                    continue
                
                # Generate MemBlock using data2Mem function
                memblock = data2Mem(data_node)
                
                # updated_data = Data(
                #     name=data_node.name,
                #     shape=data_node.shape,
                #     dtype=data_node.dtype,
                #     tags=data_node.tags,
                #     payload=data_node.payload,
                #     memref=memblock
                # )
                
                # Replace the node in the graph
                self._nodes[nid].memref = memblock
                memref_mapping[nid] = memblock
            elif node_type == "concat":
                concat_node: ConcatData = node
                if concat_node.inferred_memref is not None:
                    # Skip if inferred_memref already exists
                    memref_mapping[nid] = concat_node.inferred_memref
                    warnings.warn(f"ConcatData node '{concat_node.name}' already has an inferred_memref, skipping regeneration")
                    continue
                
                # Generate inferred memref for the concat data
                inferred_memblock = concat_node.infer_memref(self.input_pairs(nid))                    
                memref_mapping[nid] = inferred_memblock
            # elif node_type == "view":
            #     view_node: ViewData = node
            #     # ViewData nodes do not have their own memref; they reference the source Data's memref
            #     if len(self._in_edges[nid]) != 1:
            #         raise ValueError(f"ViewData node '{view_node.name}' should have exactly one predecessor Data node")
                
            #     source_nid = self._in_edges[nid][0]
            #     source_node = self._nodes[source_nid]
                
            #     if not isinstance(source_node, Data):
            #         raise ValueError(f"Predecessor of ViewData node '{view_node.name}' is not a Data node")
            #     if source_node.memref is None:
            #         print(f"Warning: Source Data node '{source_node.name}' has no memref, skipping ViewData memref generation")
            #         continue
            #     # if view_node.inferred_memref is not None:
            #     #     # Skip if inferred_memref already exists
            #     #     memref_mapping[nid] = view_node.inferred_memref
            #     #     warnings.warn(f"ViewData node '{view_node.name}' already has an inferred_memref, skipping regeneration")
            #     #     continue
            #     # Generate inferred memref for the view
            #     inferred_memblock = view_node.infer_memref(source_node)                    
            #     memref_mapping[nid] = inferred_memblock

        return memref_mapping

    def update_viewdata_memrefs(self) -> None:
        """Update inferred memrefs for all ViewData nodes based on their source Data nodes.
        
        This function traverses all ViewData nodes in the graph, retrieves the memref
        from their source Data nodes, and updates the inferred_memref field accordingly.
        
        Raises:
            ValueError: If a ViewData node's source Data node does not have a memref.
        """
        for nid, node in self._nodes.items():
            node_type = self.kind_of(nid)
            
            if node_type == "view":
                view_node: ViewData = node
                
                if len(self._in_edges[nid]) != 1:
                    raise ValueError(f"ViewData node '{view_node.name}' should have exactly one predecessor Data node")
                
                source_nid = self._in_edges[nid][0]
                source_node = self._nodes[source_nid]
                
                if not isinstance(source_node, (Data, ConcatData)):
                    raise ValueError(f"Predecessor of ViewData node '{view_node.name}' is not a Data node")
                if source_node.memref is None:
                    raise ValueError(f"Source Data node '{source_node.name}' has no memref, cannot update ViewData memref")
                
                # Update inferred memref for the view
                inferred_memblock = view_node.infer_memref(source_node)
                view_node.inferred_memref = inferred_memblock
    
    def get_core_statistics(self) -> Dict[int, Dict[str, int]]:
        """Get statistics about node distribution across cores.
        
        Returns:
            Dict[int, Dict[str, int]]: Statistics for each core including:
                - total_nodes: Total number of nodes
                - data_nodes: Number of data nodes
                - view_nodes: Number of view nodes
                - operation_nodes: Number of operation nodes
                - control_nodes: Number of control nodes
                - communication_nodes: Number of communication nodes
        """
        stats = {}
        
        for core_id in self.get_all_core_ids():
            core_nodes = self.get_nodes_by_core(core_id)
            
            data_count = 0
            view_count = 0
            operation_count = 0
            control_count = 0
            communication_count = 0
            
            for nid in core_nodes:
                node_type = self.kind_of(nid)
                if node_type == "data":
                    data_count += 1
                elif node_type == "view":
                    view_count += 1
                elif node_type == "operation":
                    operation_count += 1
                elif node_type == "control":
                    control_count += 1
                elif node_type == "communication":
                    communication_count += 1
            
            stats[core_id] = {
                "total_nodes": len(core_nodes),
                "data_nodes": data_count,
                "view_nodes": view_count,
                "operation_nodes": operation_count,
                "control_nodes": control_count,
                "communication_nodes": communication_count
            }
        
        return stats

    def gen_communication_ops(self) -> None:
        """Generate communication operations (Send/Recv) for data transfers between different cores.
        
        This function identifies data and view nodes that are produced on one core and consumed on another,
        and inserts appropriate Send and Recv operations to facilitate data transfer.
        
        Raises:
            ValueError: If a data or view node is missing core ID information.
        """
        while True:
        
            communication_resolved = True
            comm_connections: List[Tuple[Optional[NodeId], Optional[NodeId]]] = []
            
            for nid, node in self._nodes.items():
                node_already_gen = False
                node_type = self.kind_of(nid)
                
                if node_type in ("data", "view", "concat"):
                    data_node = node  # Could be Data or ViewData
                    producers = self._in_edges[nid]
                    consumers = self._out_edges[nid]
                    
                    # assert len(producers) <= 1, f"{node_type.capitalize()} node has multiple producers, which is unsupported."
                    
                    for producer in producers:  # 遍历所有产生数据的节点
                        producer_core = self.get_core_id(producer)
                        if producer_core is None:
                            raise ValueError(f"Producer node {producer} missing core ID information.")
                        
                        if producer_core != self.get_core_id(nid):
                            connection = (producer, nid)
                            if connection in comm_connections:
                                continue
                            comm_connections.append(connection)
                            communication_resolved = False
                            node_already_gen = True
                    
                    if not node_already_gen:  
                        for consumer in consumers:
                            consumer_core = self.get_core_id(consumer)
                            if consumer_core is None:
                                raise ValueError(f"Consumer node {consumer} missing core ID information.")
                            
                            if consumer_core != self.get_core_id(nid):
                                connection = (nid, consumer)
                                if connection in comm_connections:
                                    continue
                                comm_connections.append(connection)
                                communication_resolved = False
                                node_already_gen = True
                    else:  # 如果生产者已经跨核，则先忽视消费者
                        continue
            
            for src_nid, dst_nid in comm_connections:
                if src_nid is None or dst_nid is None:
                    continue
                
                src_node = self._nodes[src_nid]
                dst_node = self._nodes[dst_nid]
                src_core_id = self.get_core_id(src_nid)
                dst_core_id = self.get_core_id(dst_nid)
                
                if self.kind_of(src_nid) in ("data", "view", "concat") and self.kind_of(dst_nid) == "operation":
                    data_node = src_node  # Could be Data or ViewData
                    
                    # Create Send operation on producer core
                    send_op = SendOp(name=f"Send.{data_node.name}.core{src_core_id[0]}_{src_core_id[1]}_to_core{dst_core_id[0]}_{dst_core_id[1]}", attrs={"source": self.get_core_id(src_nid), "dest": self.get_core_id(dst_nid), "tag": self.tag_id})
                    send_nid = self.add_node(send_op, core_id=self.get_core_id(src_nid))

                    # Create Recv operation on consumer core
                    recv_op = RecvOp(name=f"Recv.{data_node.name}.core{src_core_id[0]}_{src_core_id[1]}_to_core{dst_core_id[0]}_{dst_core_id[1]}", attrs={"source": self.get_core_id(src_nid),  "tag": self.tag_id})
                    recv_nid = self.add_node(recv_op, core_id=self.get_core_id(dst_nid))
                    self.tag_id = (self.tag_id + 1) % 256
                    
                    # Copy node to consumer core (handle both Data and ViewData)
                    if self.kind_of(src_nid) == "data":
                        copied_data = Data(
                            name=f"{data_node.name}.copy.core{src_core_id[0]}_{src_core_id[1]}_to_core{dst_core_id[0]}_{dst_core_id[1]}",
                            shape=data_node.shape,
                            dtype=data_node.dtype,
                            tags=data_node.tags,
                            payload=data_node.payload,
                            memref=None   # Memref will be allocated on the consumer core
                        )
                    elif self.kind_of(src_nid) == "view":
                        view_data_node_id = self.predecessors(src_nid)[0]
                        view_data_node = self._nodes[view_data_node_id]
                        if not isinstance(view_data_node, Data):
                            raise ValueError(f"ViewData node {data_node.name} does not have a valid source Data node.")
                        copied_data = resolve_view_data(view=data_node, source_data=view_data_node, src_core_id=src_core_id, dst_core_id=dst_core_id)
                    else:  # concat node
                        concat_data_node_id = src_nid
                        concat_data_node = self._nodes[concat_data_node_id]
                        if not isinstance(concat_data_node, ConcatData):
                            raise ValueError(f"ConcatData node {data_node.name} is not a valid ConcatData node.")
                        input_data_nodes = self.input_pairs(concat_data_node_id)
                        copied_data = resolve_concat_data(concat=concat_data_node, inputs=input_data_nodes, src_core_id=src_core_id, dst_core_id=dst_core_id)
                    
                    copied_nid = self.add_node(copied_data, core_id=self.get_core_id(dst_nid))
                    
                    self.disconnect(src_nid, dst_nid)
                    # Connect Original Data/View -> Send -> Recv
                    self.connect(src_nid, send_nid)
                    self.connect(send_nid, recv_nid)
                    
                    # Connect Recv -> Copied Data/View -> Consumer Op
                    self.connect(recv_nid, copied_nid)
                    self.connect(copied_nid, dst_nid)
                
                elif self.kind_of(src_nid) == "data" and self.kind_of(dst_nid) == "concat":
                    data_node: Data = src_node
                    
                    # Create Send operation on producer core
                    send_op = SendOp(name=f"Send.{data_node.name}.core{src_core_id[0]}_{src_core_id[1]}_to_core{dst_core_id[0]}_{dst_core_id[1]}", attrs={"source": self.get_core_id(src_nid), "dest": self.get_core_id(dst_nid), "tag": self.tag_id})
                    send_nid = self.add_node(send_op, core_id=self.get_core_id(src_nid))
                    
                    # Create Recv operation on consumer core
                    recv_op = RecvOp(name=f"Recv.{data_node.name}.core{src_core_id[0]}_{src_core_id[1]}_to_core{dst_core_id[0]}_{dst_core_id[1]}", attrs={"source": self.get_core_id(src_nid), "tag": self.tag_id})
                    recv_nid = self.add_node(recv_op, core_id=self.get_core_id(dst_nid))
                    
                    self.tag_id = (self.tag_id + 1) % 256
                    
                    # Copy data node to consumer core
                    copied_data = Data(
                        name=f"{data_node.name}.copy.core{src_core_id[0]}_{src_core_id[1]}_to_core{dst_core_id[0]}_{dst_core_id[1]}",
                        shape=data_node.shape,
                        dtype=data_node.dtype,
                        tags=data_node.tags,
                        payload=data_node.payload,
                        memref=None   # Memref will be allocated on the consumer core
                    )
                    copied_nid = self.add_node(copied_data, core_id=self.get_core_id(dst_nid))
                    
                    self.disconnect(src_nid, dst_nid)
                    # Connect Original Data -> Send -> Recv
                    self.connect(src_nid, send_nid)
                    self.connect(send_nid, recv_nid)
                    
                    # Connect Recv -> Copied Data -> Concat
                    self.connect(recv_nid, copied_nid)
                    self.connect(copied_nid, dst_nid)

                elif self.kind_of(src_nid) == "data" and self.kind_of(dst_nid) == "view":
                    data_node: Data = src_node
                    
                    # Create Send operation on producer core
                    send_op = SendOp(name=f"Send.{data_node.name}.core{src_core_id[0]}_{src_core_id[1]}_to_core{dst_core_id[0]}_{dst_core_id[1]}", attrs={"source": self.get_core_id(src_nid), "dest": self.get_core_id(dst_nid), "tag": self.tag_id})
                    send_nid = self.add_node(send_op, core_id=self.get_core_id(src_nid))
                    
                    # Create Recv operation on consumer core
                    recv_op = RecvOp(name=f"Recv.{data_node.name}.core{src_core_id[0]}_{src_core_id[1]}_to_core{dst_core_id[0]}_{dst_core_id[1]}", attrs={"source": self.get_core_id(src_nid), "tag": self.tag_id})
                    recv_nid = self.add_node(recv_op, core_id=self.get_core_id(dst_nid))
                    
                    self.tag_id = (self.tag_id + 1) % 256
                    
                    # Copy data node to consumer core
                    copied_data = Data(
                        name=f"{data_node.name}.copy.core{src_core_id[0]}_{src_core_id[1]}_to_core{dst_core_id[0]}_{dst_core_id[1]}",
                        shape=data_node.shape,
                        dtype=data_node.dtype,
                        tags=data_node.tags,
                        payload=data_node.payload,
                        memref=None   # Memref will be allocated on the consumer core
                    )
                    copied_nid = self.add_node(copied_data, core_id=self.get_core_id(dst_nid))
                    
                    self.disconnect(src_nid, dst_nid)
                    # Connect Original Data -> Send -> Recv
                    self.connect(src_nid, send_nid)
                    self.connect(send_nid, recv_nid)
                    
                    # Connect Recv -> Copied Data -> View
                    self.connect(recv_nid, copied_nid)
                    self.connect(copied_nid, dst_nid)
                
                elif self.kind_of(src_nid) == "operation" and self.kind_of(dst_nid) == "data":
                    data_node: Data = dst_node
                    
                    # Create Send operation on producer core
                    send_op = SendOp(name=f"Send.{data_node.name}.core{src_core_id[0]}_{src_core_id[1]}_to_core{dst_core_id[0]}_{dst_core_id[1]}", attrs={"source": self.get_core_id(src_nid), "dest": self.get_core_id(dst_nid), "tag": self.tag_id})
                    send_nid = self.add_node(send_op, core_id=self.get_core_id(src_nid))
                    
                    # Create Recv operation on consumer core
                    recv_op = RecvOp(name=f"Recv.{data_node.name}.core{src_core_id[0]}_{src_core_id[1]}_to_core{dst_core_id[0]}_{dst_core_id[1]}", attrs={"source": self.get_core_id(src_nid), "tag": self.tag_id})
                    recv_nid = self.add_node(recv_op, core_id=self.get_core_id(dst_nid))
                    self.tag_id = (self.tag_id + 1) % 256
                    
                    
                    # Copy data node to producer core
                    copied_data = Data(
                        name=f"{data_node.name}.copy.core{src_core_id[0]}_{src_core_id[1]}_to_core{dst_core_id[0]}_{dst_core_id[1]}",
                        shape=data_node.shape,
                        dtype=data_node.dtype,
                        tags=data_node.tags,
                        payload=data_node.payload,
                        memref=None   # Memref will be allocated on the consumer core
                    )
                    copied_nid = self.add_node(copied_data, core_id=self.get_core_id(src_nid))
                    
                    self.disconnect(src_nid, dst_nid)
                    # Connect Op -> Copied Data -> Send -> Recv
                    self.connect(src_nid, copied_nid)
                    self.connect(copied_nid, send_nid)
                    self.connect(send_nid, recv_nid)
                    
                    # Connect Recv -> Original Data
                    self.connect(recv_nid, dst_nid)
                
                else:
                    raise ValueError(f"Unsupported communication pattern between {src_nid} (kind: {self.kind_of(src_nid)}) and {dst_nid} (kind: {self.kind_of(dst_nid)})")

            if communication_resolved:
                break      
        

    def to_dot(self, vertical: bool = True, show_core_ids: bool = True) -> str:
        """Generate Graphviz DOT format visualization with core ID information.
        
        Extends the base to_dot method to include core ID information in node labels.
        
        Args:
            vertical: Whether to use vertical layout.
            show_core_ids: Whether to include core ID information in node labels.
            
        Returns:
            str: DOT format string with core information.
        """
        if not show_core_ids:
            return super().to_dot(vertical)
        
        lines = ["digraph hardware_graph {"]
        rankdir = "TB" if vertical else "LR"
        lines.append(f"  rankdir={rankdir};")
        lines.append("  node [fontsize=10];")
        lines.append("  edge [fontsize=8];")
        
        # Generate nodes with core ID information
        for nid, node in self._nodes.items():
            node_type = self.kind_of(nid)
            safe_nid = nid.replace(":", "_").replace("-", "_")
            core_id = self.get_core_id(nid)
            
            if node_type == "data":
                d: Data = node
                shape_str = str(d.shape) if d.shape else "Unknown"
                dtype_str = d.dtype.value if d.dtype else "Unknown"
                memref_info = f"\\nMemRef: {d.memref.length}" if d.memref else ""
                # 添加地址信息显示
                addr_info = ""
                if d.memref and hasattr(d.memref, 'addr') and d.memref.addr is not None:
                    addr_info = f"\\nAddr: {d.memref.addr}"
                label = f"{d.name}\\nCore: {core_id}\\nShape: {shape_str}\\nType: {dtype_str}{memref_info}{addr_info}"
                
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
                # View节点：六边形，显示视图信息和核心ID
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
                    view_info = f"Slice [{v.slice_start}:{v.slice_end}] at dim{v.slice_dimension_index}"
                else:
                    view_info = f"View: {v.view_type}"
                
                # 添加inferred地址信息显示
                addr_info = ""
                if hasattr(v, 'inferred_memref') and v.inferred_memref and hasattr(v.inferred_memref, 'addr') and v.inferred_memref.addr is not None:
                    addr_info = f"\\nAddr: {v.inferred_memref.addr}"
                
                label = f"{v.name}\\nCore: {core_id}\\n{view_info}\\nShape: {shape_str}\\nType: {dtype_str}{addr_info}"
                
                lines.append(
                    f'  "{safe_nid}" [label="{label}", shape=hexagon, '
                    f'style=filled, fillcolor=lightgreen];'
                )
                
            elif node_type == "concat":
                # ConcatData 节点：双八边形，显示核心与推断的 mem 信息
                c = node  # ConcatData
                out_shape_str = str(getattr(c, 'out_shape', None)) if getattr(c, 'out_shape', None) else "Unknown"
                dtype_str = c.dtype.value if getattr(c, 'dtype', None) else "Unknown"
                # 长度与地址信息（来自 inferred_memref）
                memref_info = ""
                addr_info = ""
                inferred = getattr(c, 'inferred_memref', None)
                if inferred is not None and getattr(inferred, 'length', None) is not None:
                    memref_info = f"\\nMemRef: {inferred.length}"
                if inferred is not None and getattr(inferred, 'addr', None) is not None:
                    addr_info = f"\\nAddr: {inferred.addr}"

                label = (
                    f"{c.name}\\nCore: {core_id}\\nConcat\\nOutShape: {out_shape_str}\\nType: {dtype_str}"
                    f"{memref_info}{addr_info}"
                )
                lines.append(
                    f'  "{safe_nid}" [label="{label}", shape=doubleoctagon, '
                    f'style=filled, fillcolor=lightsalmon];'
                )
                
            elif node_type == "control":
                ctrl_name = getattr(node, "name", f"ctrl_{nid}")
                label = f"{ctrl_name}\\nCore: {core_id}"
                lines.append(
                    f'  "{safe_nid}" [label="{label}", shape=diamond, '
                    f'style=filled, fillcolor=lightgrey];'
                )
                
            elif node_type == "communication":
                comm_op = node
                comm_name = getattr(comm_op, "name", f"comm_{nid}")
                label = f"{comm_name}\\nCore: {core_id}"
                lines.append(
                    f'  "{safe_nid}" [label="{label}", shape=hexagon, '
                    f'style=filled, fillcolor=lightcoral];'
                )
                
            else:  # operation
                op_name = getattr(node, "name", f"op_{nid}")
                label = f"{op_name}\\nCore: {core_id}"
                lines.append(
                    f'  "{safe_nid}" [label="{label}", shape=box, '
                    f'style=filled, fillcolor=lightyellow];'
                )
        
        # Generate edges (same as base class)
        for src_nid in self._nodes.keys():
            safe_src = src_nid.replace(":", "_").replace("-", "_")
            
            for idx, dst_nid in enumerate(self._out_edges[src_nid]):
                safe_dst = dst_nid.replace(":", "_").replace("-", "_")
                
                src_port = self._out_ports[src_nid][idx]
                dst_port = None
                for in_idx, in_src in enumerate(self._in_edges[dst_nid]):
                    if in_src == src_nid:
                        dst_port = self._in_ports[dst_nid][in_idx]
                        break
                
                port_labels = []
                if src_port is not None:
                    port_labels.append(f"s{src_port}")
                if dst_port is not None:
                    port_labels.append(f"d{dst_port}")
                
                if port_labels:
                    label_str = f' [label="{"/".join(port_labels)}", color=blue]'
                else:
                    label_str = ' [color=blue]'
                
                lines.append(f'  "{safe_src}" -> "{safe_dst}"{label_str};')
        
        # Generate control edges
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

    def print_core_summary(self) -> None:
        """Print a summary of core assignments and statistics."""
        print("\n=== Hardware Graph Core Summary ===")
        stats = self.get_core_statistics()
        
        for core_id in sorted(stats.keys()):
            core_stats = stats[core_id]
            print(f"\nCore {core_id}:")
            print(f"  Total nodes: {core_stats['total_nodes']}")
            print(f"  Data nodes: {core_stats['data_nodes']}")
            print(f"  View nodes: {core_stats['view_nodes']}")
            print(f"  Operation nodes: {core_stats['operation_nodes']}")
            print(f"  Control nodes: {core_stats['control_nodes']}")
            print(f"  Communication nodes: {core_stats['communication_nodes']}")
            
            # List specific nodes
            core_nodes = self.get_nodes_by_core(core_id)
            if core_nodes:
                print(f"  Nodes: {', '.join(core_nodes)}")
        
        print(f"\nTotal cores used: {len(stats)}")
        print(f"Total nodes: {len(self._nodes)}")
