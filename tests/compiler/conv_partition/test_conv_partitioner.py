import unittest
import torch
import torch.nn.functional as F
import random
import sys
import os

# Add the project root to the path so we can import the partitioner
from core.compiler.op_partition.conv_partitioner import ConvPartitioner

class TestConvPartitioner(unittest.TestCase):
    
    def _run_partition_test(self, batch_size, h_in, w_in, c_in, c_out, kernel_size, stride, padding, dilation, split_config):
        """Helper function to run a single partition test and compare with PyTorch."""
        partitioner = ConvPartitioner(
            batch_size=batch_size,
            h_in=h_in, w_in=w_in, c_in=c_in, c_out=c_out,
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        
        tasks = partitioner.partition(*split_config)
        
        # Create random input and weights
        x = torch.randn(batch_size, c_in, h_in, w_in)
        weight = torch.randn(c_out, c_in, kernel_size[0], kernel_size[1])
        bias = torch.randn(c_out)
        
        # Original convolution
        x_padded_original = F.pad(x, (padding[2], padding[3], padding[0], padding[1]))
        y_original = F.conv2d(
            x_padded_original, weight, bias, 
            stride=stride, padding=0, dilation=dilation
        )
        
        # Reconstruct output from partitioned tasks
        y_reconstructed = torch.zeros_like(y_original)
        
        for task in tasks:
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
        self.assertLess(max_diff, 1e-4, f"Max difference {max_diff} exceeds tolerance for config: {split_config}")

    def test_random_convolutions(self):
        """Test randomly generated convolution configurations."""
        num_tests = 2000
        for i in range(num_tests):
            # Randomly generate convolution parameters
            batch_size = random.randint(1, 4)
            h_in = random.randint(8, 64)
            w_in = random.randint(8, 64)
            c_in = random.choice([3, 16, 32, 64])
            c_out = random.choice([16, 32, 64, 128])
            
            k = random.choice([1, 3, 5, 7])
            kernel_size = (k, k)
            
            s = random.choice([1, 2, 3])
            stride = (s, s)
            
            p = random.choice([0, 1, 2, k//2])
            padding = (p, p, p, p)
            
            d = random.choice([1, 2])
            dilation = (d, d)
            
            # Ensure kernel size with dilation is not larger than input size + padding
            effective_k = (k - 1) * d + 1
            if effective_k > h_in + 2 * p or effective_k > w_in + 2 * p:
                continue
            
            # Randomly generate split configuration
            num_h = random.randint(1, 4)
            num_w = random.randint(1, 4)
            num_c = random.randint(1, 4)
            split_config = (num_h, num_w, num_c)
            
            with self.subTest(i=i, params=(batch_size, h_in, w_in, c_in, c_out, kernel_size, stride, padding, dilation, split_config)):
                self._run_partition_test(
                    batch_size, h_in, w_in, c_in, c_out, 
                    kernel_size, stride, padding, dilation, split_config
                )

    def test_edge_cases(self):
        """Test specific edge cases."""
        edge_cases = [
            # 1x1 conv, no padding, stride 1
            (1, 16, 16, 32, 64, (1, 1), (1, 1), (0, 0, 0, 0), (1, 1), (2, 2, 2)),
            # Large padding, large stride
            (2, 32, 32, 16, 32, (5, 5), (3, 3), (4, 4, 4, 4), (1, 1), (3, 3, 1)),
            # Asymmetric padding (if supported by partitioner, currently symmetric in test but partitioner supports it)
            (1, 20, 20, 8, 16, (3, 3), (1, 1), (1, 2, 0, 1), (1, 1), (2, 1, 2)),
            # Dilation > 1
            (1, 28, 28, 16, 16, (3, 3), (1, 1), (2, 2, 2, 2), (2, 2), (2, 2, 1)),
            # Extreme split (1x1x1 output per task)
            (1, 4, 4, 4, 4, (3, 3), (1, 1), (1, 1, 1, 1), (1, 1), (4, 4, 4)),
        ]
        
        for i, params in enumerate(edge_cases):
            with self.subTest(i=i, params=params):
                self._run_partition_test(*params)

if __name__ == '__main__':
    unittest.main()