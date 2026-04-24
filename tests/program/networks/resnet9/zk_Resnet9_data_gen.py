import torch
import torch.nn as nn
import os
import math
import torchvision.transforms as transforms
from PIL import Image
import site
site.addsitedir('software/src')

from quantization_utils import *
from ppq_onnx2torch import extractInt8QuantizedOnnx
from basics.data_block import DataBlock, convertFeatureToDataBlock, calculateTensorAddressLength, toHexStr, convertTensorToDataBlock, convertWeightToDataBlock

# test_name = "resnet9"
# test_path = "./software/tests/" + test_name
# result_path = "./software/tests/" + test_name + "/results"
# os.makedirs(test_path, exist_ok=True)
# os.makedirs(result_path, exist_ok=True)
#

add_scale = 1

class QuantizedResNet9(nn.Module):
    def __init__(self, layer_info, num_classes=10, test_name=f"testcases/resnet9_single_core", core_y=0, core_x=0, addr=0x4689):
        super(QuantizedResNet9, self).__init__()
        self.test_name = test_name
        self.core_y = core_y
        self.core_x = core_x       
        self.addr = addr 
        
        self.layer_info = layer_info
        
        # input conv
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)

        # residual block 1
        self.conv2_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2_1 = nn.ReLU(inplace=True)
        
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.shortcut2 = nn.Sequential()
        self.relu2 = nn.ReLU(inplace=True)

        # residual block 2
        self.conv3_1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True)
        self.relu3_1 = nn.ReLU(inplace=True)
        
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.shortcut3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, stride=2, padding=0, bias=True),
        )
        self.relu3 = nn.ReLU(inplace=True)

        # residual block 3
        self.conv4_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.relu4_1 = nn.ReLU(inplace=True)
        
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.shortcut4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0, bias=True),
        )
        self.relu4 = nn.ReLU(inplace=True)

        self.avg_pool = nn.AvgPool2d(8)
        self.avg_pool_conv = nn.Conv2d(64, 64, kernel_size=8, stride=8, padding=0, bias=True)
        self.fc = nn.Linear(64, num_classes)
        

    def getParams(self):
        layer_info = self.layer_info
        
        self.conv1.weight.data = torch.Tensor(layer_info["/conv1/Conv"]["weight"])
        self.conv1.bias.data = torch.Tensor(layer_info["/conv1/Conv"]["bias"])
        
        self.conv2_1.weight.data = torch.Tensor(layer_info["/conv2_1/Conv"]["weight"])
        self.conv2_1.bias.data = torch.Tensor(layer_info["/conv2_1/Conv"]["bias"])
        
        self.conv2_2.weight.data = torch.Tensor(layer_info["/conv2_2/Conv"]["weight"])
        self.conv2_2.bias.data = torch.Tensor(layer_info["/conv2_2/Conv"]["bias"])
        
        self.conv3_1.weight.data = torch.Tensor(layer_info["/conv3_1/Conv"]["weight"])
        self.conv3_1.bias.data = torch.Tensor(layer_info["/conv3_1/Conv"]["bias"])
        
        self.conv3_2.weight.data = torch.Tensor(layer_info["/conv3_2/Conv"]["weight"])
        self.conv3_2.bias.data = torch.Tensor(layer_info["/conv3_2/Conv"]["bias"])
        
        self.shortcut3[0].weight.data = torch.Tensor(layer_info["/shortcut3/shortcut3.0/Conv"]["weight"])
        self.shortcut3[0].bias.data = torch.Tensor(layer_info["/shortcut3/shortcut3.0/Conv"]["bias"])
        
        self.conv4_1.weight.data = torch.Tensor(layer_info["/conv4_1/Conv"]["weight"])
        self.conv4_1.bias.data = torch.Tensor(layer_info["/conv4_1/Conv"]["bias"])
        
        self.conv4_2.weight.data = torch.Tensor(layer_info["/conv4_2/Conv"]["weight"])
        self.conv4_2.bias.data = torch.Tensor(layer_info["/conv4_2/Conv"]["bias"])
        
        self.shortcut4[0].weight.data = torch.Tensor(layer_info["/shortcut4/shortcut4.0/Conv"]["weight"])
        self.shortcut4[0].bias.data = torch.Tensor(layer_info["/shortcut4/shortcut4.0/Conv"]["bias"])
        
        self.fc.weight.data = torch.Tensor(layer_info["/fc/Gemm"]["weight"])
        self.fc.bias.data = torch.Tensor(layer_info["/fc/Gemm"]["bias"])
        
        avg_pool_weight=torch.zeros(64, 64, 8, 8)
        for i in range(64):
            avg_pool_weight[i, i, :, :] = 1
        self.avg_pool_conv.weight.data = avg_pool_weight
        self.avg_pool_conv.bias.data = torch.zeros(64)
    
    def quantize(self):
        self.conv1.weight.data = (self.conv1.weight.data / self.layer_info["/conv1/Conv"]["weight_scale"] + self.layer_info["/conv1/Conv"]["weight_zero_point"]).round().clamp(-128, 127)
        self.conv1.bias.data = (self.conv1.bias.data / self.layer_info["/conv1/Conv"]["bias_scale"] + self.layer_info["/conv1/Conv"]["bias_zero_point"]).round()
        # tensor_output(self.conv1.weight.to(torch.int8), "INT8", (2, 3, 1, 0), result_path + '/' + 'conv1_weight')
        # tensor_output(self.conv1.bias.to(torch.int32),"INT32", (0,), result_path + '/' + 'conv1_bias')
        
        self.conv2_1.weight.data = (self.conv2_1.weight.data / self.layer_info["/conv2_1/Conv"]["weight_scale"] + self.layer_info["/conv2_1/Conv"]["weight_zero_point"]).round().clamp(-128, 127)
        self.conv2_1.bias.data = (self.conv2_1.bias.data / self.layer_info["/conv2_1/Conv"]["bias_scale"] + self.layer_info["/conv2_1/Conv"]["bias_zero_point"]).round()
        # tensor_output(self.conv2_1.weight.to(torch.int8), "INT8", (2, 3, 1, 0), result_path + '/' + 'conv2_weight')
        # tensor_output(self.conv2_1.bias.to(torch.int32),"INT32", (0,), result_path + '/' + 'conv2_bias')

        self.conv2_2.weight.data = (self.conv2_2.weight.data / self.layer_info["/conv2_2/Conv"]["weight_scale"] + self.layer_info["/conv2_2/Conv"]["weight_zero_point"]).round().clamp(-128, 127)
        self.conv2_2.bias.data = (self.conv2_2.bias.data / self.layer_info["/conv2_2/Conv"]["bias_scale"] + self.layer_info["/conv2_2/Conv"]["bias_zero_point"]).round()
        # tensor_output(self.conv2_2.weight.to(torch.int8), "INT8", (2, 3, 1, 0), result_path + '/' + 'conv3_weight')
        # tensor_output(self.conv2_2.bias.to(torch.int32),"INT32", (0,), result_path + '/' + 'conv3_bias')


        self.conv3_1.weight.data = (self.conv3_1.weight.data / self.layer_info["/conv3_1/Conv"]["weight_scale"] + self.layer_info["/conv3_1/Conv"]["weight_zero_point"]).round().clamp(-128, 127)
        self.conv3_1.bias.data = (self.conv3_1.bias.data / self.layer_info["/conv3_1/Conv"]["bias_scale"] + self.layer_info["/conv3_1/Conv"]["bias_zero_point"]).round()
        # tensor_output(self.conv3_1.weight.to(torch.int8), "INT8", (2, 3, 1, 0), result_path + '/' + 'conv4_weight')
        # tensor_output(self.conv3_1.bias.to(torch.int32),"INT32", (0,), result_path + '/' + 'conv4_bias')


        self.conv3_2.weight.data = (self.conv3_2.weight.data / self.layer_info["/conv3_2/Conv"]["weight_scale"] + self.layer_info["/conv3_2/Conv"]["weight_zero_point"]).round().clamp(-128, 127)
        self.conv3_2.bias.data = (self.conv3_2.bias.data / self.layer_info["/conv3_2/Conv"]["bias_scale"] + self.layer_info["/conv3_2/Conv"]["bias_zero_point"]).round()
        # tensor_output(self.conv3_2.weight.to(torch.int8), "INT8", (2, 3, 1, 0), result_path + '/' + 'conv5_weight')
        # tensor_output(self.conv3_2.bias.to(torch.int32),"INT32", (0,), result_path + '/' + 'conv5_bias')

        self.shortcut3[0].weight.data = (self.shortcut3[0].weight.data / self.layer_info["/shortcut3/shortcut3.0/Conv"]["weight_scale"] + self.layer_info["/shortcut3/shortcut3.0/Conv"]["weight_zero_point"]).round().clamp(-128, 127)
        self.shortcut3[0].bias.data = (self.shortcut3[0].bias.data / self.layer_info["/shortcut3/shortcut3.0/Conv"]["bias_scale"] + self.layer_info["/shortcut3/shortcut3.0/Conv"]["bias_zero_point"]).round()
        # tensor_output(self.shortcut3[0].weight.data.to(torch.int8), "INT8", (2, 3, 1, 0), result_path + '/' + 'conv4e_weight')
        # tensor_output(self.shortcut3[0].bias.data.to(torch.int32),"INT32", (0,), result_path + '/' + 'conv4e_bias')
        self.conv4_1.weight.data = (self.conv4_1.weight.data / self.layer_info["/conv4_1/Conv"]["weight_scale"] + self.layer_info["/conv4_1/Conv"]["weight_zero_point"]).round().clamp(-128, 127)
        self.conv4_1.bias.data = (self.conv4_1.bias.data / self.layer_info["/conv4_1/Conv"]["bias_scale"] + self.layer_info["/conv4_1/Conv"]["bias_zero_point"]).round()
        # tensor_output(self.conv4_1.weight.data.to(torch.int8), "INT8", (2, 3, 1, 0), result_path + '/' + 'conv6_weight')
        # tensor_output(self.conv4_1.bias.data.to(torch.int32),"INT32", (0,), result_path + '/' + 'conv6_bias')

        self.conv4_2.weight.data = (self.conv4_2.weight.data / self.layer_info["/conv4_2/Conv"]["weight_scale"] + self.layer_info["/conv4_2/Conv"]["weight_zero_point"]).round().clamp(-128, 127)
        self.conv4_2.bias.data = (self.conv4_2.bias.data / self.layer_info["/conv4_2/Conv"]["bias_scale"] + self.layer_info["/conv4_2/Conv"]["bias_zero_point"]).round()
        # tensor_output(self.conv4_2.weight.data.to(torch.int8), "INT8", (2, 3, 1, 0), result_path + '/' + 'conv7_weight')
        # tensor_output(self.conv4_2.bias.data.to(torch.int32),"INT32", (0,), result_path + '/' + 'conv7_bias')
        self.shortcut4[0].weight.data = (self.shortcut4[0].weight.data / self.layer_info["/shortcut4/shortcut4.0/Conv"]["weight_scale"] + self.layer_info["/shortcut4/shortcut4.0/Conv"]["weight_zero_point"]).round().clamp(-128, 127)
        self.shortcut4[0].bias.data = (self.shortcut4[0].bias.data / self.layer_info["/shortcut4/shortcut4.0/Conv"]["bias_scale"] + self.layer_info["/shortcut4/shortcut4.0/Conv"]["bias_zero_point"]).round()
        # tensor_output(self.shortcut4[0].weight.data.to(torch.int8), "INT8", (2, 3, 1, 0), result_path + '/' + 'conv6e_weight')
        # tensor_output(self.shortcut4[0].bias.data.to(torch.int32),"INT32", (0,), result_path + '/' + 'conv6e_bias')
        self.fc.weight.data = (self.fc.weight.data / self.layer_info["/fc/Gemm"]["weight_scale"] + self.layer_info["/fc/Gemm"]["weight_zero_point"]).round().clamp(-128, 127)        
        self.fc.bias.data = (self.fc.bias.data / self.layer_info["/fc/Gemm"]["bias_scale"] + self.layer_info["/fc/Gemm"]["bias_zero_point"]).round()
        # tensor_output(self.fc.weight.data.to(torch.int8).reshape(10, 64, 1, 1), "INT8", (2, 3, 1, 0), result_path + '/' + 'fc_weight')
        # bb = torch.zeros(32)
        # bb[:10] = self.fc.bias.data
        # tensor_output(bb.to(torch.int32),"INT32", (0,), result_path + '/' + 'fc_bias')
    
    def forward(self, x):
        def write_data_to_file(data_list, core_x, core_y):
            test_name = self.test_name
            os.makedirs(test_name, exist_ok=True)
            # os.makedirs(f"{test_name}", exist_ok=True)

            data_list.sort(key=lambda x: x[1])

            with open(f"{test_name}/{core_y}_{core_x}_mem_output.txt", "w") as f:
                for data_inst in data_list:
                    data, data_addr = data_inst[0], data_inst[1]
                    for i in range(len(data)):
                        addr_hex = f"{(data_addr+i):04x}"
                        data_hex = data[i]
                        f.write(f"@{addr_hex} {data_hex}\n")
        
        def write_lvds_to_file(data_list):
            test_name = self.test_name
            os.makedirs(test_name, exist_ok=True)
            # os.makedirs(f"{test_name}", exist_ok=True)

            data_list.sort(key=lambda x: x[1])

            with open(f"{test_name}/lvds_golden.txt", "w") as f:
                for data_inst in data_list:
                    data, data_addr = data_inst[0], data_inst[1]
                    for i in range(len(data)):
                        addr_hex = f"{((data_addr+i)<<2)+0:04x}"
                        data_hex = data[i][64:0]
                        f.write(f"@{addr_hex} {data_hex}\n")
                        
                        addr_hex = f"{((data_addr+i)<<2)+1:04x}"
                        data_hex = data[i][128:64]
                        f.write(f"@{addr_hex} {data_hex}\n")
                        
                        addr_hex = f"{((data_addr+i)<<2)+2:04x}"
                        data_hex = data[i][192:128]
                        f.write(f"@{addr_hex} {data_hex}\n")
                        
                        addr_hex = f"{((data_addr+i)<<2)+3:04x}"
                        data_hex = data[i][256:192]
                        f.write(f"@{addr_hex} {data_hex}\n")
        
        # no bn
        # input conv
        x0 = (x / self.layer_info["/conv1/Conv"]["input_scale"]).round().clamp(-128, 127)
        # 在此处导出输入量化后的数据，注意这里的输入也是深层卷积，不是浅层卷积

        # tensor_output(x, "INT8", (0, 2, 3, 1), result_path + '/' + 'conv1_in')
        
        
        x = self.conv1(x0)
        # 注意此处在芯片上已经变成int8
        # 在代码中，后面做了两次cut，即residual2和x，这两次cut的scale不同
        # 在芯片配置时应该先按照后一次x的scale配置，再用lut之类的方法得到residual2的值（因为后一次精度更高）
        # tensor_output(x, "INT8", (0, 2, 3, 1), result_path + '/' + 'conv1_out')
        
        x = x.to(torch.bfloat16)
        
        conv1_w_all = torch.zeros((32, 32, 3, 3), dtype=torch.int8)
        conv1_w_all[:, 0:self.conv1.weight.data.shape[1], :, :] = self.conv1.weight.data
        # write_data_to_file([
        #     # [convertTensorToDataBlock(x0.view(1, 3, 32, 32), "INT8", (0,2,3,1), addressing="32B"), 256],
        #     # [convertTensorToDataBlock(conv1_w_all.view(32, 32, 3, 3), "INT8", (2,3,1,0), addressing="32B"), 1280],
        #     [convertTensorToDataBlock(x.view(1, 32, 32, 32), "BF16", (0,2,3,1), addressing="32B"), 1572],
        # ], core_y=0, core_x=0)
        # exit()        
        
        x = self.relu1(x)
        
        
        # residual block 1
        # residual2 = cut_scale(x, self.layer_info["/conv1/Conv"]["bias_scale"], self.layer_info["/Add"]["input_1_scale"])
        # residual2 = cut_scale(x, self.layer_info["/conv1/Conv"]["bias_scale"] * add_scale, self.layer_info["/Add"]["input_1_scale"])
        residual2 = x * self.layer_info["/conv1/Conv"]["bias_scale"] * add_scale / self.layer_info["/Add"]["input_1_scale"]
        
        # write_data_to_file([
        #     [convertTensorToDataBlock(residual2.view(1, 32, 32, 32), "INT8", (0,2,3,1), addressing="32B"), 3620],
        # ], core_y=0, core_x=0)
        # exit()
        
        residual2 = self.shortcut2(residual2)
        # tensor_output(residual2, "INT8", (0, 2, 3, 1), result_path + '/' + 'conv1e_out')
        x = cut_scale(x, self.layer_info["/conv1/Conv"]["bias_scale"], self.layer_info["/conv2_1/Conv"]["input_scale"])
        # print(self.layer_info["/conv1/Conv"]["bias_scale"]/self.layer_info["/conv2_1/Conv"]["input_scale"])
        # tensor_output(x, "INT8", (0, 2, 3, 1), result_path + '/' + 'conv1_relu')
        
        # write_data_to_file([
        #     [convertTensorToDataBlock(x.view(1, 32, 32, 32), "INT8", (0,2,3,1), addressing="32B"), 4644],
        # ], core_y=0, core_x=0)
        # exit()
        x = x.to(torch.float32)
        x = self.conv2_1(x)
        x = x.to(torch.bfloat16)
        x = cut_scale(x, self.layer_info["/conv2_1/Conv"]["bias_scale"], self.layer_info["/conv2_2/Conv"]["input_scale"])
               
        
        x = self.relu2_1(x)        

        x = x.to(torch.float32)
        x = self.conv2_2(x)
        x = x.to(torch.bfloat16)
        
        # x = cut_scale(x, self.layer_info["/conv2_2/Conv"]["bias_scale"], self.layer_info["/Add"]["input_2_scale"])
        # x = cut_scale(x, self.layer_info["/conv2_2/Conv"]["bias_scale"] * add_scale, self.layer_info["/Add"]["input_2_scale"])
        x = x * self.layer_info["/conv2_2/Conv"]["bias_scale"] * add_scale / self.layer_info["/Add"]["input_2_scale"]
        
        # write_data_to_file([
        #     [convertTensorToDataBlock(residual2.view(1, 32, 32, 32), "BF16", (0,2,3,1), addressing="32B"), 4213],
        #     [convertTensorToDataBlock(x.view(1, 32, 32, 32), "BF16", (0,2,3,1), addressing="32B"), 9333],
        # ], core_y=0, core_x=0)
        # exit()
        
        x = x + residual2
        # x = cut_scale(x, self.layer_info["/Add"]["input_1_scale"], self.layer_info["/Add"]["output_scale"])
        x = cut_scale(x, self.layer_info["/Add"]["input_1_scale"] / add_scale, self.layer_info["/Add"]["output_scale"])
        
        x = self.relu2(x)
        
        # write_data_to_file([
        #     [convertTensorToDataBlock(x.view(1, 32, 32, 32), "INT8", (0,2,3,1), addressing="32B"), 4214],
        # ], core_y=0, core_x=0)
        # exit()

        # residual block 2
        
        x = x.to(torch.float32)
        
        residual3 = self.shortcut3(x)
        residual3 = residual3.to(torch.int32)
        residual3 = residual3.to(torch.bfloat16)
        # residual3 = cut_scale(residual3, self.layer_info["/shortcut3/shortcut3.0/Conv"]["bias_scale"], self.layer_info["/Add_1"]["input_1_scale"])
        # residual3 = cut_scale(residual3, self.layer_info["/shortcut3/shortcut3.0/Conv"]["bias_scale"] * add_scale, self.layer_info["/Add_1"]["input_1_scale"])
        residual3 = residual3 * (self.layer_info["/shortcut3/shortcut3.0/Conv"]["bias_scale"] * add_scale / self.layer_info["/Add_1"]["input_1_scale"])
        
        # write_data_to_file([
        #     [convertTensorToDataBlock(residual3.view(1, 32, 16, 16), "BF16", (0,2,3,1), addressing="32B"), 2716],
        # ], core_y=0, core_x=0)
        # exit()
        
        
        # tensor_output(residual3, "INT8", (0, 2, 3, 1), result_path + '/' + 'conv4e_out')
        x = self.conv3_1(x)
        x = x.to(torch.bfloat16)
        # # tensor_output(x, "INT8", (0, 2, 3, 1), result_path + '/' + 'conv4_out')
        # 注意此处在芯片上已经变成int8
        x = self.relu3_1(x)
        x = cut_scale(x, self.layer_info["/conv3_1/Conv"]["bias_scale"], self.layer_info["/conv3_2/Conv"]["input_scale"])
        
        # write_data_to_file([
        #     [convertTensorToDataBlock(residual3.view(1, 32, 16, 16), "BF16", (0,2,3,1), addressing="32B"), 3523],
        #     [convertTensorToDataBlock(x.view(1, 32, 16, 16), "INT8", (0,2,3,1), addressing="32B"), 4035],
        # ], core_y=0, core_x=0)
        # exit()
        
        
        # tensor_output(x, "INT8", (0, 2, 3, 1), result_path + '/' + 'conv4_relu')
        x = self.conv3_2(x)
        x = x.to(torch.int32)
        x = x.to(torch.bfloat16)
        # x = cut_scale(x, self.layer_info["/conv3_2/Conv"]["bias_scale"], self.layer_info["/Add_1"]["input_2_scale"])
        # x = cut_scale(x, self.layer_info["/conv3_2/Conv"]["bias_scale"] * add_scale, self.layer_info["/Add_1"]["input_2_scale"])
        x = x * (self.layer_info["/conv3_2/Conv"]["bias_scale"] * add_scale / self.layer_info["/Add_1"]["input_2_scale"])
        
        x = x + residual3
        # x = cut_scale(x, self.layer_info["/Add_1"]["input_1_scale"], self.layer_info["/Add_1"]["output_scale"])
        
        # write_data_to_file([
        #     [convertTensorToDataBlock(x.view(1, 32, 16, 16), "BF16", (0,2,3,1), addressing="32B"), 2793],
        # ], core_y=0, core_x=0)
        # exit()
        
        x = cut_scale(x, self.layer_info["/Add_1"]["input_1_scale"] / add_scale, self.layer_info["/Add_1"]["output_scale"])
        x = self.relu3(x)
        
        # write_data_to_file([
        #     [convertTensorToDataBlock(x.view(1, 32, 16, 16), "INT8", (0,2,3,1), addressing="32B"), 3307],
        # ], core_y=0, core_x=0)
        # exit()
        
        # tensor_output(x, "INT8", (0, 2, 3, 1), result_path + '/' + 'conv5_relu')
        # residual block 3
        residual4 = self.shortcut4(x)
        residual4 = residual4.to(torch.int32)
        residual4 = residual4.to(torch.bfloat16)
        # residual4 = cut_scale(residual4, self.layer_info["/shortcut4/shortcut4.0/Conv"]["bias_scale"], self.layer_info["/Add_2"]["input_1_scale"])
        # residual4 = cut_scale(residual4, self.layer_info["/shortcut4/shortcut4.0/Conv"]["bias_scale"] * add_scale, self.layer_info["/Add_2"]["input_1_scale"])
        residual4 = residual4 * self.layer_info["/shortcut4/shortcut4.0/Conv"]["bias_scale"] * add_scale / self.layer_info["/Add_2"]["input_1_scale"]
        
        # write_data_to_file([
        #     [convertTensorToDataBlock(residual4.view(1, 64, 8, 8), "BF16", (0,2,3,1), addressing="32B"), 5385],
        # ], core_y=0, core_x=0)
        # exit()
        
        x = self.conv4_1(x)
        x = x.to(torch.int32)
        x = x.to(torch.bfloat16)
        
        # write_data_to_file([
        #     [convertTensorToDataBlock(x.view(1, 64, 8, 8), "BF16", (0,2,3,1), addressing="32B"), 3455],
        #     # [convertTensorToDataBlock(x.view(1, 64, 8, 8), "INT8", (0,2,3,1), addressing="32B"), 5641],
        # ], core_y=0, core_x=0)
        # exit()
       
        x = cut_scale(x, self.layer_info["/conv4_1/Conv"]["bias_scale"], self.layer_info["/conv4_2/Conv"]["input_scale"])
        
        # tensor_output(x, "INT8", (0, 2, 3, 1), result_path + '/' + 'conv6_out')
        
        x = self.relu4_1(x)
        
        # write_data_to_file([
        #     [convertTensorToDataBlock(x.view(1, 64, 8, 8), "INT8", (0,2,3,1), addressing="32B"), 4223],
        #     # [convertTensorToDataBlock(x.view(1, 64, 8, 8), "INT8", (0,2,3,1), addressing="32B"), 3455],
        # ], core_y=0, core_x=0)
        # exit()
        
        
        x = self.conv4_2(x)
        x = x.to(torch.int32)
        x = x.to(torch.bfloat16)
        
        # write_data_to_file([
        #     [convertTensorToDataBlock(x.view(1, 64, 8, 8), "BF16", (0,2,3,1), addressing="32B"), 4873],
        # ], core_y=0, core_x=0)
        # exit()
        
        # x = cut_scale(x, self.layer_info["/conv4_2/Conv"]["bias_scale"], self.layer_info["/Add_2"]["input_2_scale"])
        # x = cut_scale(x, self.layer_info["/conv4_2/Conv"]["bias_scale"] * add_scale, self.layer_info["/Add_2"]["input_2_scale"])
        x = x * self.layer_info["/conv4_2/Conv"]["bias_scale"] * add_scale / self.layer_info["/Add_2"]["input_2_scale"]
        
        # tensor_output(x, "INT8", (0, 2, 3, 1), result_path + '/' + 'conv7_out')
        x = x + residual4
        # x = cut_scale(x, self.layer_info["/Add_2"]["input_1_scale"], self.layer_info["/Add_2"]["output_scale"])
        # x = cut_scale(x, self.layer_info["/Add_2"]["input_1_scale"] / add_scale, self.layer_info["/Add_2"]["output_scale"])
        x = x * self.layer_info["/Add_2"]["input_1_scale"] / add_scale / self.layer_info["/Add_2"]["output_scale"]
        
        x = self.relu4(x)
        
        # write_data_to_file([
        #     [convertTensorToDataBlock(x.view(1, 64, 8, 8), "BF16", (0,2,3,1), addressing="32B"), 4874],
        # ], core_y=0, core_x=0)
        # exit()
        
        x = x.to(torch.bfloat16)
        x = self.avg_pool(x)
        
        x = cut_scale(x, self.layer_info["/Add_2"]["output_scale"], self.layer_info["/fc/Gemm"]["input_scale"])
        # x = x * self.layer_info["/Add_2"]["output_scale"] / self.layer_info["/fc/Gemm"]["input_scale"]    
        
        # write_data_to_file([
        #     [convertTensorToDataBlock(x.view(1, 64, 1, 1), "INT8", (0,2,3,1), addressing="32B"), 4619],
        # ], core_y=0, core_x=0)
        # exit()  
        
        # x = self.avg_pool_conv(x)
        # x = cut_scale(x, self.layer_info["/Add_2"]["output_scale"] / 64, self.layer_info["/fc/Gemm"]["input_scale"])
        # tensor_output(x, "INT8", (0, 2, 3, 1), result_path + '/' + 'avg_pool_out')   
             
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        x = x.to(torch.int32)
        x = x.to(torch.bfloat16)
        x = cut_scale(x, self.layer_info["/fc/Gemm"]["bias_scale"], self.layer_info["/fc/Gemm"]["output_scale"])
        # tensor_output(x, "INT8", (0, 1), result_path + '/' + 'fc_out')
        
        # write_data_to_file([
        #     [convertTensorToDataBlock(x.view(1, 10, 1, 1), "INT8", (0,2,3,1), addressing="32B"), self.addr],
        # ], core_y=self.core_y, core_x=self.core_x)
        
        write_lvds_to_file([
            [convertTensorToDataBlock(x.view(1, 10, 1, 1), "INT8", (0,2,3,1), addressing="32B"), 0],
        ])
        
        # x = x * self.layer_info["/fc/Gemm"]["output_scale"]
        
        return x

if __name__ == "__main__":
    # 读取图像

    net = QuantizedResNet9(extractInt8QuantizedOnnx("test_gen/network/resnet9/MQuantized.onnx"))
    net.getParams()
    net.quantize()

    
    import matplotlib.pyplot as plt
    
    # input_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    input_class = ['truck']
    
    for c in input_class:
        image = torch.load(f"test_gen/network/resnet9/input_data/{c}.tensor")

        image_show = image.permute(0, 2, 3, 1).numpy().squeeze(0)

        # 对数据进行归一化处理，如果需要 (可选)
        # 一般图像的值范围为 0-1 或 0-255，这里假设你要把随机生成的数值归一化到 0-1
        image_show = (image_show - image_show.min()) / (image_show.max() - image_show.min())

        # # 使用 matplotlib 显示图像
        # plt.imshow(image_show)
        # plt.axis('off')  # 去掉坐标轴
        # plt.show()

        # conv1e_weight=torch.zeros(32, 32, 1, 1)
        # for i in range(32):
        #     conv1e_weight[i, i, 0, 0] = 1
        # # tensor_output(conv1e_weight, "INT8", (2, 3, 1, 0), result_path + '/' + 'conv1e_weight')
        # conv1e_bias=torch.zeros(32)
        # # tensor_output(conv1e_bias, "INT32", (0,), result_path + '/' + 'conv1e_bias')

        # avg_pool_weight=torch.zeros(64, 64, 8, 8)
        # for i in range(64):
        #     avg_pool_weight[i, i, :, :] = 1
        # # tensor_output(avg_pool_weight, "INT8", (2, 3, 1, 0), result_path + '/' + 'avg_pool_weight')
        # avg_pool_bias=torch.zeros(64)
        # # tensor_output(avg_pool_bias, "INT32", (0,), result_path + '/' + 'avg_pool_bias')

        
        res = net(image)
        print(res)
        print(res.argmax(1))
