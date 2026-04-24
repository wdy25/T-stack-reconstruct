import math
from typing import List, Dict, Tuple, Any
import torch
import torch.nn.functional as F

class ConvPartitioner:
    """
    A lightweight partitioner for Convolution operators.
    It splits the convolution workload along the spatial dimensions (H, W) 
    and the output channel dimension (C_out).
    """
    def __init__(
        self,
        batch_size: int,
        h_in: int,
        w_in: int,
        c_in: int,
        c_out: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int, int, int] = (0, 0, 0, 0), # top, bottom, left, right
        dilation: Tuple[int, int] = (1, 1)
    ):
        self.batch_size = batch_size
        self.h_in = h_in
        self.w_in = w_in
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_h, self.kernel_w = kernel_size
        self.stride_h, self.stride_w = stride
        self.pad_top, self.pad_bottom, self.pad_left, self.pad_right = padding
        self.dilation_h, self.dilation_w = dilation
        
        # Calculate output dimensions
        self.h_out = (self.h_in + self.pad_top + self.pad_bottom - self.dilation_h * (self.kernel_h - 1) - 1) // self.stride_h + 1
        self.w_out = (self.w_in + self.pad_left + self.pad_right - self.dilation_w * (self.kernel_w - 1) - 1) // self.stride_w + 1

    def get_valid_splits(self, max_cores: int, min_c_out: int = 32, min_h_out: int = 1, min_w_out: int = 1) -> List[Tuple[int, int, int]]:
        """
        Search for valid split configurations (num_h, num_w, num_c) 
        that fit within max_cores and satisfy minimum granularity constraints.
        
        Args:
            max_cores: Maximum number of cores available.
            min_c_out: Minimum number of output channels per split.
            min_h_out: Minimum spatial height per split.
            min_w_out: Minimum spatial width per split.
            
        Returns:
            A list of valid (num_h, num_w, num_c) tuples.
        """
        valid_splits = []
        for num_c in range(1, self.c_out + 1):
            if math.ceil(self.c_out / num_c) < min_c_out and num_c > 1:
                continue
            for num_h in range(1, self.h_out + 1):
                if math.ceil(self.h_out / num_h) < min_h_out and num_h > 1:
                    continue
                for num_w in range(1, self.w_out + 1):
                    if math.ceil(self.w_out / num_w) < min_w_out and num_w > 1:
                        continue
                    
                    if num_h * num_w * num_c <= max_cores:
                        valid_splits.append((num_h, num_w, num_c))
        return valid_splits

    def _get_1d_slice(self, dim_size: int, num_splits: int, split_idx: int) -> Tuple[int, int]:
        """Helper to get start and end index for a 1D split."""
        chunk_size = math.ceil(dim_size / num_splits)
        start = split_idx * chunk_size
        end = min(start + chunk_size, dim_size)
        return start, end

    def partition(self, num_h: int, num_w: int, num_c: int) -> List[Dict[str, Any]]:
        """
        Partition the convolution into num_h * num_w * num_c tasks.
        
        Returns:
            A list of tasks, where each task contains the physical slices for 
            input, output, weight, bias, and the local padding required.
        """
        tasks = []
        for c_idx in range(num_c):
            c_out_start, c_out_end = self._get_1d_slice(self.c_out, num_c, c_idx)
            if c_out_start >= c_out_end: continue
            
            for h_idx in range(num_h):
                h_out_start, h_out_end = self._get_1d_slice(self.h_out, num_h, h_idx)
                if h_out_start >= h_out_end: continue
                
                # Calculate virtual input H range (including padding)
                v_in_h_start = h_out_start * self.stride_h
                v_in_h_end = (h_out_end - 1) * self.stride_h + self.dilation_h * (self.kernel_h - 1)
                
                # Calculate physical input H range (excluding padding)
                p_in_h_start = max(0, v_in_h_start - self.pad_top)
                p_in_h_end = min(self.h_in - 1, v_in_h_end - self.pad_top)
                
                # Calculate local padding for this specific split
                local_pad_top = max(0, self.pad_top - v_in_h_start)
                local_pad_bottom = max(0, v_in_h_end - (self.h_in + self.pad_top - 1))
                
                # Adjust physical input range if it's completely outside the input tensor
                if p_in_h_start > p_in_h_end:
                    p_in_h_start = 0
                    p_in_h_end = -1
                
                for w_idx in range(num_w):
                    w_out_start, w_out_end = self._get_1d_slice(self.w_out, num_w, w_idx)
                    if w_out_start >= w_out_end: continue
                    
                    # Calculate virtual input W range (including padding)
                    v_in_w_start = w_out_start * self.stride_w
                    v_in_w_end = (w_out_end - 1) * self.stride_w + self.dilation_w * (self.kernel_w - 1)
                    
                    # Calculate physical input W range (excluding padding)
                    p_in_w_start = max(0, v_in_w_start - self.pad_left)
                    p_in_w_end = min(self.w_in - 1, v_in_w_end - self.pad_left)
                    
                    # Calculate local padding for this specific split
                    local_pad_left = max(0, self.pad_left - v_in_w_start)
                    local_pad_right = max(0, v_in_w_end - (self.w_in + self.pad_left - 1))
                    
                    # Adjust physical input range if it's completely outside the input tensor
                    if p_in_w_start > p_in_w_end:
                        p_in_w_start = 0
                        p_in_w_end = -1
                    
                    task = {
                        "task_id": (h_idx, w_idx, c_idx),
                        "output_slice": {
                            "b": (0, self.batch_size),
                            "h": (h_out_start, h_out_end),
                            "w": (w_out_start, w_out_end),
                            "c": (c_out_start, c_out_end)
                        },
                        "input_slice": {
                            "b": (0, self.batch_size),
                            "h": (p_in_h_start, p_in_h_end + 1),
                            "w": (p_in_w_start, p_in_w_end + 1),
                            "c": (0, self.c_in)
                        },
                        "weight_slice": {
                            "kh": (0, self.kernel_h),
                            "kw": (0, self.kernel_w),
                            "cin": (0, self.c_in),
                            "cout": (c_out_start, c_out_end)
                        },
                        "bias_slice": {
                            "cout": (c_out_start, c_out_end)
                        },
                        "local_padding": (local_pad_top, local_pad_bottom, local_pad_left, local_pad_right),
                        "task_shape": {
                            "output": (self.batch_size, h_out_end - h_out_start, w_out_end - w_out_start, c_out_end - c_out_start),
                            "input": (self.batch_size, p_in_h_end + 1 - p_in_h_start, p_in_w_end + 1 - p_in_w_start, self.c_in),
                            "weight": (self.kernel_h, self.kernel_w, self.c_in, c_out_end - c_out_start)
                        }
                    }
                    tasks.append(task)
                    
        return tasks

def main():
    print("=== Conv Partitioner Example ===")
    
    # Parameters similar to a typical deep_conv
    batch_size = 1
    h_in, w_in, c_in = 16, 16, 64
    c_out = 128
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (1, 1, 1, 1) # top, bottom, left, right
    dilation = (1, 1)
    
    partitioner = ConvPartitioner(
        batch_size=batch_size,
        h_in=h_in, w_in=w_in, c_in=c_in, c_out=c_out,
        kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
    )
    
    print(f"Original Input Shape: ({batch_size}, {h_in}, {w_in}, {c_in})")
    print(f"Original Output Shape: ({batch_size}, {partitioner.h_out}, {partitioner.w_out}, {c_out})")
    
    # 1. Explore valid splits for a many-core architecture (e.g., max 16 cores)
    max_cores = 16
    min_c_out = 32
    min_h_out = 2 # Spatial split shouldn't be too small
    min_w_out = 2
    
    valid_splits = partitioner.get_valid_splits(
        max_cores=max_cores, 
        min_c_out=min_c_out, 
        min_h_out=min_h_out, 
        min_w_out=min_w_out
    )
    
    print(f"\nValid split configurations (num_h, num_w, num_c) for max {max_cores} cores:")
    for split in valid_splits:
        print(f"  - {split} (Total tasks: {split[0]*split[1]*split[2]})")
        
    # 2. Choose a specific split to partition
    # Let's say we choose (num_h=2, num_w=2, num_c=2) -> 8 cores
    chosen_split = (1, 5, 3)
    print(f"\nPartitioning with chosen split: {chosen_split}")
    
    tasks = partitioner.partition(*chosen_split)
    
    for i, task in enumerate(tasks):
        print(f"\n--- Task {i} (Index: {task['task_id']}) ---")
        print(f"Output Slice (B, H, W, C): {task['output_slice']}")
        print(f"Input Slice  (B, H, W, C): {task['input_slice']}")
        print(f"Weight Slice (Kh, Kw, Cin, Cout): {task['weight_slice']}")
        print(f"Local Padding (Top, Bottom, Left, Right): {task['local_padding']}")
        print(f"Task Output Shape: {task['task_shape']['output']}")
        print(f"Task Input Shape:  {task['task_shape']['input']}")

    # 3. Functional Test with PyTorch
    print("\n=== Functional Test with PyTorch ===")
    
    # Create random input and weights
    x = torch.randn(batch_size, c_in, h_in, w_in)
    weight = torch.randn(c_out, c_in, kernel_size[0], kernel_size[1])
    bias = torch.randn(c_out)
    
    # Original convolution
    # F.conv2d expects padding as (pad_h, pad_w) if symmetric, or we can pad manually first
    x_padded_original = F.pad(x, (padding[2], padding[3], padding[0], padding[1]))
    y_original = F.conv2d(
        x_padded_original, weight, bias, 
        stride=stride, padding=0, dilation=dilation
    )
    
    # Reconstruct output from partitioned tasks
    y_reconstructed = torch.zeros_like(y_original)
    
    for task in tasks:
        # Extract slices
        out_slice = task['output_slice']
        in_slice = task['input_slice']
        w_slice = task['weight_slice']
        b_slice = task['bias_slice']
        local_pad = task['local_padding']
        
        # Get input chunk
        x_chunk = x[
            in_slice['b'][0]:in_slice['b'][1],
            in_slice['c'][0]:in_slice['c'][1],
            in_slice['h'][0]:in_slice['h'][1],
            in_slice['w'][0]:in_slice['w'][1]
        ]
        
        # Apply local padding
        # F.pad expects padding from last dimension to first: (left, right, top, bottom)
        x_chunk_padded = F.pad(
            x_chunk, 
            (local_pad[2], local_pad[3], local_pad[0], local_pad[1])
        )
        
        # Get weight and bias chunks
        w_chunk = weight[
            w_slice['cout'][0]:w_slice['cout'][1],
            w_slice['cin'][0]:w_slice['cin'][1],
            w_slice['kh'][0]:w_slice['kh'][1],
            w_slice['kw'][0]:w_slice['kw'][1]
        ]
        b_chunk = bias[b_slice['cout'][0]:b_slice['cout'][1]]
        
        # Perform local convolution
        # Note: padding is 0 here because we already applied local padding
        y_chunk = F.conv2d(
            x_chunk_padded, w_chunk, b_chunk,
            stride=stride, padding=0, dilation=dilation
        )
        
        # Place chunk into reconstructed output
        y_reconstructed[
            out_slice['b'][0]:out_slice['b'][1],
            out_slice['c'][0]:out_slice['c'][1],
            out_slice['h'][0]:out_slice['h'][1],
            out_slice['w'][0]:out_slice['w'][1]
        ] = y_chunk
        
    # Compare results
    max_diff = torch.max(torch.abs(y_original - y_reconstructed)).item()
    print(f"Max difference between original and reconstructed output: {max_diff}")
    if max_diff < 1e-4: # Float32 precision tolerance
        print("Test PASSED: Partitioned convolution matches original convolution!")
    else:
        print("Test FAILED: Results do not match.")

if __name__ == "__main__":
    main()
