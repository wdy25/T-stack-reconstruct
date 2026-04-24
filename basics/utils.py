import torch
import struct
import os
import numpy as np
import random
import torch
import torch.nn as nn
import yaml


def makeDir(filepath):
    filepath = filepath.strip()  # 去除首位空格
    parent_path, _ = os.path.split(filepath)
    isExists = os.path.exists(parent_path)  # 判断路径是否存在，存在则返回true
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(parent_path)

def load_config_from_yaml(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            config = yaml.load(file, Loader=yaml.Loader)
        return config

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 设置PyTorch在CUDA上生成随机数的种子。
    torch.cuda.manual_seed_all(seed)  # 在多GPU环境中，设置所有可见的CUDA设备的随机种子。
    torch.backends.cudnn.deterministic = True  # 设置使用CuDNN时，使得卷积算法保持确定性。这是为了确保在使用GPU加速时，卷积操作的结果是确定性的。
    torch.backends.cudnn.benchmark = False  # 关闭CuDNN的自动调整，确保在不同批次上卷积的性能一致。这也是为了保证实验的可重复性。
    torch.backends.cudnn.enabled = False  # 禁用CuDNN。这是因为CuDNN的实现可能会引入一些非确定性的因素，而在某些情况下禁用它可以获得可重复的结果。

    # 这段代码是用来设置PyTorch和其他相关库的随机种子（random
    # seed）的。在机器学习中，设置随机种子是为了使实验可重复，即每次运行代码时得到的随机结果相同，这样有助于调试和比较不同模型或算法的性能。
    # 具体来说，这段代码做了以下几个操作：
    # random.seed(seed)：设置Python的random模块的种子，确保在使用random模块生成随机数时，得到的结果是可复现的。
    # os.environ['PYTHONHASHSEED'] = str(seed)：设置Python中hash的种子，这同样有助于使得使用哈希的操作变得可重复。
    # np.random.seed(seed)：设置NumPy库的随机种子，以确保NumPy生成的随机数也是可重复的。
    # torch.manual_seed(seed)：设置PyTorch的随机种子，包括CPU上的随机数生成。
    # torch.cuda.manual_seed(seed)：

    # 总体来说，这些设置是为了确保在使用PyTorch进行深度学习实验时，通过控制随机性使实验结果具有可重复性。这在实验、调试和结果比较时都是非常重要的。


# Convert a number to hex
# BF16, FP32, INT8, INT32 can be convert to hex one by one
# BIN can be convert to hex 4 bits (MSB at index 3) at a time
def num_to_hex(num, type):
    # Convert the float to bytes
    if (type == "FP32"):
        num_bytes = struct.pack('f', num.to(torch.float32))
    elif (type == "BF16"):
        num_bytes = struct.pack('f', num.to(torch.float32))
    elif (type == "INT8"):
        num_bytes = struct.pack('b', num.to(torch.int8))
    elif (type == "INT32"):
        num_bytes = struct.pack('i', num.to(torch.int32))

    # Convert the bytes to an integer
    if (type == "BIN"):
        num_int = num[0] * 1 + num[1] * 2 + num[2] * 4 + num[3] * 8
    else:
        num_int = int.from_bytes(num_bytes, byteorder='little', signed=False)

    # output = hex(num_int)

    if (type == "BF16"):
        output = "{:08x}".format(num_int)
        output = output[:-4]
    elif (type == "FP32"):
        output = "{:08x}".format(num_int)
    elif (type == "INT8"):
        output = "{:02x}".format(num_int)
    elif (type == "INT32"):
        output = "{:08x}".format(num_int)
    elif (type == "BIN"):
        output = "{:01x}".format(int(num_int))

    return output