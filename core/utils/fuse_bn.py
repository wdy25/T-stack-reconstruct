import torch
import torch.nn as nn


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    mu = bn.running_mean
    var = bn.running_var
    gamma = bn.weight.data
    beta = bn.bias.data
    
    a = gamma / torch.sqrt(var + bn.eps)
    
    new_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups, bias=True)
    
    new_conv.weight.data = conv.weight.data * a.view(-1, 1, 1, 1)
    if conv.bias is not None:
        new_conv.bias.data = (conv.bias.data - mu) * a + beta
    else:
        new_conv.bias.data = beta - mu * a
    
    return new_conv


if __name__ == "__main__":
    conv = nn.Conv2d(3, 3, 3)
    print(conv.weight.data)
    bn = nn.BatchNorm2d(3)
    fused_conv = fuse_conv_bn(conv, bn)
    print(fused_conv.weight.data)