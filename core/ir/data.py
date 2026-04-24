from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from enum import Enum
from math import ceil
import torch
from myhdl import bin, intbv
import warnings
import struct
import numpy as np

class DataType(Enum):
    """Data types for data nodes."""

    INT8 = "int8"
    BF16 = "bf16"
    SPIKE = "spike"
    INT32 = "int32"


class MemBlock:
    """In-memory data block with payload."""

    def __init__(
        self,
        length: int,
        addr: int = -1,
        alignment: int = 32,  # in bytes
        payload: Optional[List[intbv]] = None,
    ):
        self.length = length
        self.addr = addr
        self.alignment = alignment
        self.payload = payload


@dataclass(frozen=False)
class Data:
    """Data node metadata (no payload)."""

    name: str
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[DataType] = None
    tags: Optional[Dict[str, Any]] = None
    payload: Optional[Union[torch.Tensor, str]] = None  # payload也可以是str
    memref: Optional[MemBlock] = None
    
    def __post_init__(self):
        if self.shape is not None:
            if not all(isinstance(dim, int) and dim > 0 for dim in self.shape):
                raise ValueError(f"All dimensions in shape must be positive integers, got {self.shape}")
        if self.payload is not None:
            if isinstance(self.payload, torch.Tensor):
                if self.shape is not None and tuple(self.payload.size()) != self.shape:
                    raise ValueError(f"Payload shape {tuple(self.payload.size())} does not match specified shape {self.shape}")
                if self.dtype is not None:
                    dtype_map = {
                        DataType.INT8: torch.int8,
                        DataType.BF16: torch.bfloat16,
                        DataType.SPIKE: torch.bool,  # Assuming spike data is stored as uint8
                        DataType.INT32: torch.int32,
                    }
                    expected_torch_dtype = dtype_map.get(self.dtype)
                    if expected_torch_dtype is None:
                        raise ValueError(f"Unsupported DataType: {self.dtype}")
                    if self.payload.dtype != expected_torch_dtype:
                        raise ValueError(f"Payload dtype {self.payload.dtype} does not match specified dtype {self.dtype}")
            elif isinstance(self.payload, str):
                assert self.payload == "random"
                self.payload = None
                self.gen_payload()
            else:
                raise ValueError(f"Payload must be None, torch.Tensor or str, got {type(self.payload)}")
    
    def gen_payload(self):
        if self.payload is None and self.shape is not None:
            if self.dtype is None:
                raise ValueError("Cannot generate payload without specified dtype")
            if self.dtype == DataType.INT8:
                self.payload = torch.randint(-128, 127, self.shape, dtype=torch.int8)
            elif self.dtype == DataType.BF16:
                self.payload = torch.randn(self.shape).to(torch.bfloat16)
            elif self.dtype == DataType.SPIKE:
                self.payload = torch.randint(0, 2, self.shape, dtype=torch.uint8)  # Binary spikes
            elif self.dtype == DataType.INT32:
                self.payload = torch.randint(-2147483648, 2147483647, self.shape, dtype=torch.int32)
            else:
                raise ValueError(f"Unsupported DataType: {self.dtype}")     


@dataclass(frozen=False)
class ViewData:
    """View节点：代表Data节点的一种"视角"或一部分
    
    View节点具有以下特性：
    1. 以Data节点作为输入源
    2. 可以作为Operation的输入
    3. 不包含实际数据，而是对源数据的引用
    4. 支持三种视图模式：变形视图、索引视图、变形后再索引视图
    
    变形视图（reshape view）：
    - 可以与源Data节点有不同的形状
    - 元素总数必须相同
    - 代表同一份数据的不同排列方式
    
    索引视图（slice view）：
    - 代表源Data节点的一个连续子集
    - 必须是在内存中连续的部分
    - 要满足连续要求，最后一个维度必须保持不变，如果有一个维度发生改变，则它之后的维度都必须保持不变，而它之前的索引维度必须变为1
    """
    
    name: str
    view_type: str    # "reshape"、"slice" 或 "reshape_and_slice"
    source_data: str = None  # 源Data节点的名称
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[DataType] = None
    tags: Optional[Dict[str, Any]] = None
    inferred_memref: Optional[MemBlock] = None  # 推断的内存引用
    memref: Optional[MemBlock] = None
    
    # 变形视图参数（当view_type="reshape"或"reshape_and_slice"时使用）
    target_shape: Optional[Tuple[int, ...]] = None
    
    # 索引视图参数（当view_type="slice"时使用）
    pre_idx: Optional[Iterable[int]] = None  # 索引维度之前的维度上的索引（每个维度只能有一个索引）
    slice_dimension_index: Optional[int] = None  # 需要索引的维度索引（从0开始计数）
    slice_start: Optional[int] = None  # 在索引维度上的起始索引
    slice_end: Optional[int] = None    # 在索引维度上的结束索引（不包含）
    
    def __post_init__(self):
        """初始化后的验证"""
        if self.view_type not in ["reshape", "slice", "reshape_and_slice"]:
            raise ValueError(f"view_type must be 'reshape', 'slice' or 'reshape_and_slice', got {self.view_type}")
        
        if self.view_type == "reshape" and self.target_shape is None:
            raise ValueError("reshape view requires target_shape parameter")
            
        if self.view_type == "slice":
            if not len(self.pre_idx or []) == self.slice_dimension_index:
                raise ValueError("pre_idx length must match slice_dimension_index")
            if self.slice_dimension_index is None:
                raise ValueError("slice view requires slice_dimension_index parameter")
            if self.slice_start is None or self.slice_end is None:
                raise ValueError("slice view requires slice_start and slice_end parameters")
            if self.slice_start >= self.slice_end:
                raise ValueError("slice_start must be less than slice_end")
        
        if self.view_type == "reshape_and_slice":
            if self.target_shape is None:
                raise ValueError("reshape_and_slice view requires target_shape parameter")
            if not len(self.pre_idx or []) == self.slice_dimension_index:
                raise ValueError("pre_idx length must match slice_dimension_index")
            if self.slice_dimension_index is None:
                raise ValueError("reshape_and_slice view requires slice_dimension_index parameter")
            if self.slice_start is None or self.slice_end is None:
                raise ValueError("reshape_and_slice view requires slice_start and slice_end parameters")
            if self.slice_start >= self.slice_end:
                raise ValueError("slice_start must be less than slice_end")
    
    def validate_with_source(self, source_data: Data) -> None:
        """验证View节点与源Data节点的兼容性
        
        Args:
            source_data: 源Data节点
            
        Raises:
            ValueError: 当View配置与源数据不兼容时
        """
        if source_data.shape is None:
            raise ValueError(f"Source data {self.source_data} must have a defined shape")
        
        if self.view_type == "reshape":
            self._validate_reshape_view(source_data)
        elif self.view_type == "slice":
            self._validate_slice_view(source_data)
        elif self.view_type == "reshape_and_slice":
            self._validate_reshape_and_slice_view(source_data)
    
    def _validate_reshape_view(self, source_data: Data) -> None:
        """验证变形视图的有效性"""
        if self.target_shape is None:
            raise ValueError("target_shape is required for reshape view")
        
        element_per_cell = 0
        if source_data.dtype == DataType.INT8:
            element_per_cell = 32
        elif source_data.dtype == DataType.BF16:
            element_per_cell = 16
        elif source_data.dtype == DataType.SPIKE:
            element_per_cell = 256
        elif source_data.dtype == DataType.INT32:
            element_per_cell = 8
        else:
            raise ValueError(f"Unsupported data type: {source_data.dtype}")
        
        source_size_in_cells = ceil(source_data.shape[-1] / element_per_cell)
        target_size_in_cells = ceil(self.target_shape[-1] / element_per_cell)
        
        if source_size_in_cells == target_size_in_cells:    
            assert source_data.shape[-1] <= self.target_shape[-1], "最后一维只允许补零变大，不能变小"
            # 计算源数据和目标形状的总元素数
            source_size = 1
            for dim in source_data.shape[: -1]:
                source_size *= dim
            
            target_size = 1
            for dim in self.target_shape[: -1]:
                target_size *= dim
            
            if source_size != target_size:
                raise ValueError(
                    f"Reshape view: source cell num {source_size} != target cell num {target_size}. "
                    f"Source shape: {source_data.shape}, target shape: {self.target_shape}"
                )
        else:
            assert source_data.shape[-1] % element_per_cell == 0, "最后一维必须是完整的cell"
            assert self.target_shape[-1] % element_per_cell == 0, "最后一维必须是完整的cell"
            
            source_size = 1
            for dim in source_data.shape:
                source_size *= dim
            
            target_size = 1
            for dim in self.target_shape:
                target_size *= dim
            
            if source_size != target_size:
                raise ValueError(
                    f"Reshape view: source element num {source_size} != target element num {target_size}. "
                    f"Source shape: {source_data.shape}, target shape: {self.target_shape}"
                )
        
        # 数据类型必须一致
        if self.dtype is not None and self.dtype != source_data.dtype:
            raise ValueError(
                f"Reshape view: dtype mismatch. Source: {source_data.dtype}, view: {self.dtype}"
            )
    
    def _validate_slice_view(self, source_data: Data) -> None:
        """验证索引视图的有效性"""
        if len(source_data.shape) < 2:
            raise ValueError(
                f"Slice view requires at least 2D source data, got shape: {source_data.shape}"
            )
        
        # 检查索引范围（倒数第二个维度）
        assert 0 <= self.slice_dimension_index < len(source_data.shape) - 1, "slice_index out of range"

        slice_dim = source_data.shape[self.slice_dimension_index]
        if self.slice_start < 0 or self.slice_end > slice_dim:
            raise ValueError(
                f"Slice indices [{self.slice_start}:{self.slice_end}] out of range "
                f"for dimension of size {slice_dim}"
            )
        
        # 数据类型必须一致
        if self.dtype is not None and self.dtype != source_data.dtype:
            raise ValueError(
                f"Slice view: dtype mismatch. Source: {source_data.dtype}, view: {self.dtype}"
            )
        
        # 计算切片后的形状
        expected_shape = list(source_data.shape)
        expected_shape[self.slice_dimension_index] = self.slice_end - self.slice_start
        for i in range(self.slice_dimension_index):
            expected_shape[i] = 1
        expected_shape = tuple(expected_shape)
        
        if self.shape is not None and self.shape != expected_shape:
            raise ValueError(
                f"Slice view: shape mismatch. Expected: {expected_shape}, got: {self.shape}"
            )

    def _validate_reshape_and_slice_view(self, source_data: Data) -> None:
        """验证先变形再索引视图的有效性"""
        if self.target_shape is None:
            raise ValueError("target_shape is required for reshape_and_slice view")

        element_per_cell = 0
        if source_data.dtype == DataType.INT8:
            element_per_cell = 32
        elif source_data.dtype == DataType.BF16:
            element_per_cell = 16
        elif source_data.dtype == DataType.SPIKE:
            element_per_cell = 256
        elif source_data.dtype == DataType.INT32:
            element_per_cell = 8
        else:
            raise ValueError(f"Unsupported data type: {source_data.dtype}")
        
        source_size_in_cells = ceil(source_data.shape[-1] / element_per_cell)
        target_size_in_cells = ceil(self.target_shape[-1] / element_per_cell)
        
        if source_size_in_cells == target_size_in_cells:    
            assert source_data.shape[-1] <= self.target_shape[-1], "最后一维只允许补零变大，不能变小"
            # 计算源数据和目标形状的总元素数
            source_size = 1
            for dim in source_data.shape[: -1]:
                source_size *= dim
            
            target_size = 1
            for dim in self.target_shape[: -1]:
                target_size *= dim
            
            if source_size != target_size:
                raise ValueError(
                    f"Reshape view: source cell num {source_size} != target cell num {target_size}. "
                    f"Source shape: {source_data.shape}, target shape: {self.target_shape}"
                )
        else:
            assert source_data.shape[-1] % element_per_cell == 0, "最后一维必须是完整的cell"
            assert self.target_shape[-1] % element_per_cell == 0, "最后一维必须是完整的cell"
            
            source_size = 1
            for dim in source_data.shape:
                source_size *= dim
            
            target_size = 1
            for dim in self.target_shape:
                target_size *= dim
            
            if source_size != target_size:
                raise ValueError(
                    f"Reshape view: source element num {source_size} != target element num {target_size}. "
                    f"Source shape: {source_data.shape}, target shape: {self.target_shape}"
                )

        # 数据类型必须一致
        if self.dtype is not None and self.dtype != source_data.dtype:
            raise ValueError(
                f"Reshape_and_slice view: dtype mismatch. Source: {source_data.dtype}, view: {self.dtype}"
            )

        # 在目标形状上进行切片的约束检查
        reshaped_shape = self.target_shape
        if len(reshaped_shape) < 2:
            raise ValueError(
                f"Reshape_and_slice requires at least 2D target shape, got shape: {reshaped_shape}"
            )
        assert 0 <= self.slice_dimension_index < len(reshaped_shape) - 1, "slice_index out of range"

        slice_dim = reshaped_shape[self.slice_dimension_index]
        if self.slice_start < 0 or self.slice_end > slice_dim:
            raise ValueError(
                f"Slice indices [{self.slice_start}:{self.slice_end}] out of range "
                f"for dimension of size {slice_dim}"
            )

        # 计算切片后的期望形状
        expected_shape = list(reshaped_shape)
        expected_shape[self.slice_dimension_index] = self.slice_end - self.slice_start
        for i in range(self.slice_dimension_index):
            expected_shape[i] = 1
        expected_shape = tuple(expected_shape)

        if self.shape is not None and self.shape != expected_shape:
            raise ValueError(
                f"Reshape_and_slice view: shape mismatch. Expected: {expected_shape}, got: {self.shape}"
            )
    
    def infer_shape_and_dtype(self, source_data: Data) -> Tuple[Tuple[int, ...], DataType]:
        """根据源数据推断View的形状和数据类型
        
        Args:
            source_data: 源Data节点
            
        Returns:
            Tuple[Tuple[int, ...], DataType]: (推断的形状, 数据类型)
        """
        if source_data.shape is None or source_data.dtype is None:
            raise ValueError("Source data must have defined shape and dtype")
        
        if self.view_type == "reshape":
            if self.target_shape is None:
                raise ValueError("target_shape is required for reshape view")
            self._validate_reshape_view(source_data)
            self.source_data = source_data.name
            self.shape = self.target_shape
            self.dtype = source_data.dtype
        
        elif self.view_type == "slice":
            if self.slice_dimension_index is None or self.slice_start is None or self.slice_end is None:
                raise ValueError("slice_index, slice_start, and slice_end are required for slice view")
            self._validate_slice_view(source_data)
            # 计算切片后的形状
            expected_shape = list(source_data.shape)
            expected_shape[self.slice_dimension_index] = self.slice_end - self.slice_start
            for i in range(self.slice_dimension_index):
                expected_shape[i] = 1
            expected_shape = tuple(expected_shape)
            self.source_data = source_data.name
            self.shape = expected_shape
            self.dtype = source_data.dtype
        
        elif self.view_type == "reshape_and_slice":
            if self.target_shape is None:
                raise ValueError("target_shape is required for reshape_and_slice view")
            if self.slice_dimension_index is None or self.slice_start is None or self.slice_end is None:
                raise ValueError("slice_index, slice_start, and slice_end are required for reshape_and_slice view")
            self._validate_reshape_and_slice_view(source_data)
            # 计算在变形后进行切片的形状
            expected_shape = list(self.target_shape)
            expected_shape[self.slice_dimension_index] = self.slice_end - self.slice_start
            for i in range(self.slice_dimension_index):
                expected_shape[i] = 1
            expected_shape = tuple(expected_shape)
            self.source_data = source_data.name
            self.shape = expected_shape
            self.dtype = source_data.dtype
        return self.shape, self.dtype
    
    def infer_memref(self, source_data: Data) -> MemBlock:
        """推断View的内存引用
        
        Args:
            source_data: 源Data节点
            
        Returns:
            MemBlock: 推断的内存引用
        """
        if source_data.memref is None:
            raise ValueError(f"Source data {source_data.name} must have a memref to infer view memref")
        
        # 计算View的内存块
        view_mem = data2Mem(Data(
            name=self.name,
            shape=self.shape,
            dtype=self.dtype,
            tags=self.tags,
            payload=None,  # View节点不包含实际数据
            memref=None
        ))
        
        if source_data.memref.addr is None or source_data.memref.addr < 0:
            view_mem.addr = source_data.memref.addr
        # 根据源数据的内存地址调整View的内存地址
        elif self.view_type == "slice":
            # 计算切片在内存中的偏移量
            element_per_cell = 0
            if source_data.dtype == DataType.INT8:
                element_per_cell = 32
            elif source_data.dtype == DataType.BF16:
                element_per_cell = 16
            elif source_data.dtype == DataType.SPIKE:
                element_per_cell = 256
            elif source_data.dtype == DataType.INT32:
                element_per_cell = 8
            else:
                raise ValueError(f"Unsupported data type: {source_data.dtype}")
            
            inner_cells = ceil(source_data.shape[-1] / element_per_cell)
            offset_idx = []
            offset_gap = []
            for i, idx in enumerate(self.pre_idx or []):
                offset_idx.append(idx)
                gap = 1
                for j in range(i + 1, len(source_data.shape) - 1):
                    gap *= source_data.shape[j]
                gap *= inner_cells
                offset_gap.append(gap)
            offset_idx.append(self.slice_start)
            gap = 1
            for j in range(self.slice_dimension_index + 1, len(source_data.shape) - 1):
                gap *= source_data.shape[j]
            gap *= inner_cells
            offset_gap.append(gap)
            offset = 0
            for i in range(len(offset_idx)):
                offset += offset_idx[i] * offset_gap[i]
            view_mem.addr = source_data.memref.addr + offset
        elif self.view_type == "reshape_and_slice":
            # 在变形的形状上进行切片的地址偏移计算（内存布局与原始相同）
            element_per_cell = 0
            if source_data.dtype == DataType.INT8:
                element_per_cell = 32
            elif source_data.dtype == DataType.BF16:
                element_per_cell = 16
            elif source_data.dtype == DataType.SPIKE:
                element_per_cell = 256
            elif source_data.dtype == DataType.INT32:
                element_per_cell = 8
            else:
                raise ValueError(f"Unsupported data type: {source_data.dtype}")

            reshaped_shape = self.target_shape
            inner_cells = ceil(reshaped_shape[-1] / element_per_cell)
            offset_idx = []
            offset_gap = []
            for i, idx in enumerate(self.pre_idx or []):
                offset_idx.append(idx)
                gap = 1
                for j in range(i + 1, len(reshaped_shape) - 1):
                    gap *= reshaped_shape[j]
                gap *= inner_cells
                offset_gap.append(gap)
            offset_idx.append(self.slice_start)
            gap = 1
            for j in range(self.slice_dimension_index + 1, len(reshaped_shape) - 1):
                gap *= reshaped_shape[j]
            gap *= inner_cells
            offset_gap.append(gap)
            offset = 0
            for i in range(len(offset_idx)):
                offset += offset_idx[i] * offset_gap[i]
            view_mem.addr = source_data.memref.addr + offset
        else:
            view_mem.addr = source_data.memref.addr
        
        self.inferred_memref = view_mem
        self.memref = view_mem
        return view_mem


def data2Mem(data: Data) -> MemBlock:
    """Convert a Data node to a MemBlock."""
    assert len(data.shape) >= 1
    cell = 0
    if data.dtype == DataType.INT8:
        cell = 32.0
    elif data.dtype == DataType.BF16:
        cell = 16.0
    elif data.dtype == DataType.SPIKE:
        cell = 256.0
    elif data.dtype == DataType.INT32:
        cell = 8.0
    else:
        raise ValueError(f"Unsupported data type: {data.dtype}")
    
    cell_in_vector = ceil(data.shape[-1] / cell)
    length = cell_in_vector
    for i in range(len(data.shape) - 1):
        length *= data.shape[i]
    
    if data.payload is not None:
        if isinstance(data.payload, torch.Tensor):
            assert data.payload.size() == torch.Size(data.shape), "Payload shape does not match data shape."
            payload = convertTensorToMem(data.payload, data.dtype)
        else:
            raise ValueError("Payload must be a torch.Tensor.")
    else:
        payload = None

    return MemBlock(length=length, payload=payload, addr=data.memref.addr if data.memref else -1)


def create_reshape_view(name: str, target_shape: Tuple[int, ...], 
                       tags: Optional[Dict[str, Any]] = None) -> ViewData:
    """创建变形视图节点
    
    变形视图允许以不同的形状查看同一份数据，但总元素数必须保持一致。
    这类似于NumPy的reshape操作，但不会复制数据。
    
    Args:
        name: View节点的名称
        source_data_name: 源Data节点的名称
        target_shape: 目标形状，总元素数必须与源数据相同
        tags: 可选的标签字典
    
    Returns:
        ViewData: 创建的变形视图节点
    
    Example:
        # 将 (2, 3, 4) 的数据变形为 (6, 4)
        view = create_reshape_view("reshaped_data", "original_data", (6, 4))
    """
    return ViewData(
        name=name,
        view_type="reshape",
        target_shape=target_shape,
        tags=tags
    )


def create_slice_view(name: str, pre_idx: List[int], slice_idx: int, slice_start: int, slice_end: int,
                     tags: Optional[Dict[str, Any]] = None) -> ViewData:
    """创建索引视图节点
    
    索引视图代表源数据在倒数第二个维度上的一个连续子集。
    这种限制确保了在内存中的连续性，便于硬件加速器访问。
    
    Args:
        name: View节点的名称
        source_data_name: 源Data节点的名称
        slice_start: 在倒数第二个维度上的起始索引（包含）
        slice_end: 在倒数第二个维度上的结束索引（不包含）
        tags: 可选的标签字典
    
    Returns:
        ViewData: 创建的索引视图节点
    
    Example:
        # 从形状为 (batch, 128, features) 的数据中选择 [32:96] 的子集
        # 结果形状为 (batch, 64, features)
        view = create_slice_view("sliced_data", "original_data", 32, 96)
    """
    return ViewData(
        name=name,
        view_type="slice",
        pre_idx=pre_idx,
        slice_dimension_index=slice_idx,
        slice_start=slice_start,
        slice_end=slice_end,
        tags=tags
    )


def create_reshape_and_slice_view(
    name: str,
    target_shape: Tuple[int, ...],
    pre_idx: List[int],
    slice_idx: int,
    slice_start: int,
    slice_end: int,
    tags: Optional[Dict[str, Any]] = None,
) -> ViewData:
    """创建先变形再索引的视图节点

    先将源数据以 target_shape 进行变形（不改变内存布局），随后在变形后的形状上按照给定
    维度与范围进行切片。该操作不复制数据，仅改变视角。

    Args:
        name: View 节点名称
        target_shape: 先变形到的目标形状（元素总数需与源数据一致）
        pre_idx: 索引维度之前各维的单点索引，长度必须等于 slice_idx
        slice_idx: 在目标形状中的索引维度（0-based），必须小于 len(target_shape)-1
        slice_start: 该维度上的起始索引（包含）
        slice_end: 该维度上的结束索引（不包含）
        tags: 可选标签

    Returns:
        ViewData: 创建的视图节点
    """
    return ViewData(
        name=name,
        view_type="reshape_and_slice",
        target_shape=target_shape,
        pre_idx=pre_idx,
        slice_dimension_index=slice_idx,
        slice_start=slice_start,
        slice_end=slice_end,
        tags=tags,
    )


def resolve_view_data(view: ViewData, source_data: Data, src_core_id: List, dst_core_id: List) -> Data:
    """将ViewData解析为具体的Data节点
    
    根据View的配置和源数据，生成一个新的Data节点，
    该节点具有正确的形状和数据类型信息。
    
    Args:
        view: 要解析的ViewData节点
        source_data: 源Data节点
    
    Returns:
        Data: 解析后的Data节点，包含正确的形状和类型信息
    
    Raises:
        ValueError: 当View配置与源数据不兼容时
    """
    # 验证View与源数据的兼容性
    view.validate_with_source(source_data)
    
    # 推断形状和数据类型
    inferred_shape, inferred_dtype = view.infer_shape_and_dtype(source_data)
    
    new_payload = None
    if source_data.payload is not None:
        if isinstance(source_data.payload, torch.Tensor):
            # 根据View的配置提取相应的子集或变形数据
            if view.view_type == "reshape":
                new_payload = source_data.payload.reshape(inferred_shape)
            elif view.view_type == "slice":
                # 构建切片对象
                slice_obj = [slice(None)] * len(source_data.shape)
                for i, idx in enumerate(view.pre_idx or []):
                    slice_obj[i] = slice(idx, idx+1)
                slice_obj[view.slice_dimension_index] = slice(view.slice_start, view.slice_end)
                new_payload = source_data.payload[tuple(slice_obj)]
            elif view.view_type == "reshape_and_slice":
                # 先 reshape 再 slice
                reshaped = source_data.payload.reshape(view.target_shape)
                slice_obj = [slice(None)] * len(view.target_shape)
                for i, idx in enumerate(view.pre_idx or []):
                    slice_obj[i] = slice(idx, idx+1)
                slice_obj[view.slice_dimension_index] = slice(view.slice_start, view.slice_end)
                new_payload = reshaped[tuple(slice_obj)]
    
    # 创建新的Data节点
    return Data(
        name=f"{view.name}.copy.core{src_core_id[0]}_{src_core_id[1]}_to_core{dst_core_id[0]}_{dst_core_id[1]}",
        shape=inferred_shape,
        dtype=inferred_dtype,
        tags=view.tags,
        payload=new_payload,  # View节点不包含实际数据
        memref=None    # 内存引用将在需要时计算
    )


def convertTensorToMem(x: torch.Tensor, type: DataType, addressing: str="32B") -> List[intbv]:
    if addressing != "32B":
        raise NotImplementedError("Not implemented yet")

    # 将输入张量转换为指定的数据类型
    if type == DataType.BF16:
        if x.dtype != torch.bfloat16:
            warnings.warn("Converting input tensor to BF16")
            x = x.to(torch.bfloat16)
    elif type == DataType.INT8:
        if x.dtype != torch.int8:
            warnings.warn("Converting input tensor to INT8")
            x = x.to(torch.int8)
    elif type == DataType.INT32:
        if x.dtype != torch.int32:
            warnings.warn("Converting input tensor to INT32")
            x = x.to(torch.int32)
    elif type == DataType.SPIKE:
        if x.dtype != torch.bool:
            warnings.warn("Converting input tensor to BIN")
            x = x.to(torch.bool)

    # 不改变维度顺序（如需可在外部预处理）
    permuted_x = x

    # 每32B单元包含的元素数量
    if type == DataType.INT8:
        number_in_32B = 32.0
    elif type == DataType.BF16:
        number_in_32B = 16.0
    elif type == DataType.SPIKE:
        number_in_32B = 256.0
    elif type == DataType.INT32:
        number_in_32B = 8.0
    else:
        raise ValueError(f"Unsupported data type: {type}")

    # 计算内层维度填充到 32B 单元
    inner_dimension = permuted_x.shape[-1]
    dimension_padded_to_cell = int(np.ceil(inner_dimension / number_in_32B) * number_in_32B)
    pad_len = dimension_padded_to_cell - inner_dimension
    if pad_len > 0:
        zero_to_full_cell = torch.zeros(
            size=[dim for dim in permuted_x.shape[:-1]] + [pad_len],
            dtype=permuted_x.dtype,
            device=permuted_x.device,
        )
        permuted_x = torch.cat((permuted_x, zero_to_full_cell), dim=-1)

    # 展平并按 32B 单元重排
    flat_x = torch.flatten(permuted_x)
    if type not in (DataType.BF16, DataType.SPIKE):
        flat_x = flat_x.to(torch.int32)
    length = flat_x.shape[0]
    x_in_mem = torch.reshape(flat_x, (int(length / number_in_32B), int(number_in_32B)))

    mem_list: List[intbv] = []
    for cell in x_in_mem:
        cell_str = ""
        if type == DataType.SPIKE:
            for ii in range(int(len(cell) / 4)):
                cell_str = num_to_hex(cell[ii * 4 : ii * 4 + 4], type) + cell_str
        else:
            for element in cell:
                cell_str = num_to_hex(element, type) + cell_str
        intbv_line = intbv(int(cell_str, 16), min=0, max=(1 << 256))
        mem_list.append(intbv_line)
    return mem_list


# Convert a number to hex
# BF16, FP32, INT8, INT32 can be convert to hex one by one
# BIN can be convert to hex 4 bits (MSB at index 3) at a time
def num_to_hex(num, type: DataType) -> str:
    # 将张量标量安全转为 Python 标量
    def to_python_scalar(v, as_type: str):
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu()
            if as_type == 'f':
                return float(v.to(torch.float32).item())
            elif as_type == 'b':
                return int(v.to(torch.int8).item())
            elif as_type == 'i':
                return int(v.to(torch.int32).item())
            else:
                return int(v.item())
        return v

    if type == DataType.SPIKE:
        # num 为长度为4的序列/张量，按 bit 聚合
        def bit_val(x):
            if isinstance(x, torch.Tensor):
                return int(x.detach().cpu().to(torch.int32).item())
            return int(x)
        num_int = bit_val(num[0]) * 1 + bit_val(num[1]) * 2 + bit_val(num[2]) * 4 + bit_val(num[3]) * 8
        return "{:01x}".format(int(num_int))

    # 其他类型先转换为 bytes，再转为整型
    if type == DataType.BF16:
        py_val = to_python_scalar(num, 'f')
        num_bytes = struct.pack('f', py_val)
        num_int = int.from_bytes(num_bytes, byteorder='little', signed=False)
        output = "{:08x}".format(num_int)
        return output[:-4]  # 取高 16 位
    elif type == DataType.INT8:
        py_val = to_python_scalar(num, 'b')
        num_bytes = struct.pack('b', py_val)
        num_int = int.from_bytes(num_bytes, byteorder='little', signed=False)
        return "{:02x}".format(num_int)
    elif type == DataType.INT32:
        py_val = to_python_scalar(num, 'i')
        num_bytes = struct.pack('i', py_val)
        num_int = int.from_bytes(num_bytes, byteorder='little', signed=False)
        return "{:08x}".format(num_int)
    else:
        raise ValueError(f"Unsupported data type: {type}")

def elements_to_32b_cell(elements: int, dtype: Optional[DataType]) -> int:
    """Convert element count to number of 32-byte cells for the given dtype."""
    per_cell = {
        DataType.INT32: 8,
        DataType.BF16: 16,
        DataType.INT8: 32,
        DataType.SPIKE: 256,
    }.get(dtype)
    if per_cell is None:
        raise ValueError(f"Unsupported input dtype for pooling: {dtype}.")
    return ceil(elements / per_cell)


@dataclass(frozen=False)
class ConcatData:
    """数据拼接算子（仅用于内存布局与形状校验，不做实际计算）

    语义：
    - 将多个输入 Data（端口顺序 0..N-1）在内存上顺序拼接，要求它们在分配地址时相邻。
    - 输出作为一个新的 Data 节点供后续计算视为整体数据使用。

    规则：
    1) 拼接顺序按照端口顺序（port 从小到大）。
    2) 所有输入数据 dtype 必须一致。
    3) 形状检查：将每个输入按其 dtype 的 32B 单元进行逻辑重排为二维 (R_i, C)，
       其中 C 等于该 dtype 在 32B 中可容纳的元素个数（即 per_cell），R_i 为该张量占用的 32B 单元总数。
       拼接后得到二维形状 (sum_i R_i, C)，该二维形状必须与指定的 out_shape 完全一致。

    注：本算子不修改/复制 payload，仅提供一个输出 Data 的元信息和期望的内存长度；
        当所有输入均已分配地址且相邻时，可推导输出的起始地址为第一个输入的地址。
    """

    name: str
    num_inputs: int
    shape: Tuple[int]
    inferred_memref: Optional[MemBlock] = None  # 推断的内存引用
    memref: Optional[MemBlock] = None
    dtype: Optional[DataType] = None
    # 记录端口到输入 Data 的映射（按端口顺序）
    inputs: Optional[Dict[int, Data]] = None

    def _per_cell(self, dtype: DataType) -> int:
        m = {
            DataType.INT32: 8,
            DataType.BF16: 16,
            DataType.INT8: 32,
            DataType.SPIKE: 256,
        }
        if dtype not in m:
            raise ValueError(f"Unsupported dtype for concat: {dtype}")
        return m[dtype]

    def _rows_for_shape(self, shape: Tuple[int, ...], dtype: DataType) -> int:
        if shape is None or len(shape) < 1:
            raise ValueError("Each input to ConcatData must have a defined, non-empty shape.")
        per_cell = self._per_cell(dtype)
        inner = shape[-1]
        outer = 1
        for d in shape[:-1]:
            outer *= d
        # 该张量占用的 32B 单元数
        return outer * ceil(inner / per_cell)

    def validate(self, inputs: Dict[int, Data]) -> Tuple[DataType, int, int]:
        # 数量与端口完整性
        if inputs is None or len(inputs) != self.num_inputs:
            raise ValueError(f"ConcatData expects {self.num_inputs} inputs, got {0 if inputs is None else len(inputs)}")
        for p in range(self.num_inputs):
            if p not in inputs:
                raise ValueError(f"Missing input on port {p} for ConcatData '{self.name}'.")

        # dtype一致性
        dtypes = [inputs[p].dtype for p in range(self.num_inputs)]
        if any(dt is None for dt in dtypes):
            raise ValueError("All inputs of ConcatData must have dtype defined.")
        first_dtype = dtypes[0]
        if any(dt != first_dtype for dt in dtypes):
            raise ValueError(f"All inputs of ConcatData must share the same dtype. Got: {dtypes}")
        if self.dtype is not None and self.dtype != first_dtype:
            raise ValueError(f"ConcatData dtype mismatch: specified {self.dtype} vs inputs {first_dtype}")
        dtype = first_dtype
        if self.dtype is None:
            self.dtype = dtype

        # 形状 -> 二维(R_i, C) 检查
        per_cell = self._per_cell(dtype)
        total_rows = 0
        for p in range(self.num_inputs):
            data = inputs[p]
            if data.shape is None:
                raise ValueError(f"Input on port {p} has undefined shape.")
            rows = self._rows_for_shape(data.shape, dtype)
            total_rows += rows

        output_rows = self._rows_for_shape(self.shape, dtype)
        if output_rows != total_rows:
            raise ValueError(
                f"ConcatData output rows mismatch: expected {output_rows}, but sum of inputs is {total_rows}."
            )

    def infer_memref(self, inputs: Dict[int, Data]):
        # 推断拼接后的内存长度
        self.validate(inputs)
        total_length = 0
        for p in range(self.num_inputs):
            data = inputs[p]
            mem = data2Mem(data)
            total_length += mem.length
        mem = MemBlock(length=total_length, payload=None, addr=None)
        self.inferred_memref = mem
        self.memref = mem
        
    def allocate_memref(self, inputs: Dict[int, Data]):
        # 根据分配的拼接后内存，分配输入数据的内存地址
        if self.inferred_memref is None or self.inferred_memref.addr is None or self.inferred_memref.addr < 0:
            raise ValueError("ConcatData must have a valid inferred_memref with address before allocating inputs.")
        current_addr = self.inferred_memref.addr
        for p in range(self.num_inputs):
            data = inputs[p]
            data.memref.addr = current_addr
            current_addr += data2Mem(data).length 


def resolve_concat_data(concat: ConcatData, inputs: Dict[int, Data], src_core_id: List, dst_core_id: List) -> Data:
    """将ConcatData解析为具体的Data节点
    
    根据ConcatData的配置和输入数据，生成一个新的Data节点，
    该节点具有正确的形状和数据类型信息。
    
    Args:
        concat: 要解析的ConcatData节点
        inputs: 输入Data节点的字典，键为端口号
        
    Returns:
        Data: 解析后的Data节点，包含正确的形状和类型信息
    
    Raises:
        ValueError: 当ConcatData配置与输入数据不兼容时
    """
    # 验证ConcatData与输入数据的兼容性
    concat.validate(inputs)
    
    # 推断内存引用
    concat.infer_memref(inputs)
    
    # 创建新的Data节点
    return Data(
        name=f"{concat.name}.concat.core{src_core_id[0]}_{src_core_id[1]}_to_core{dst_core_id[0]}_{dst_core_id[1]}",
        shape=concat.shape,
        dtype=concat.dtype,
        tags={},
        payload=None,  # ConcatData节点不包含实际数据
        memref=None
    )