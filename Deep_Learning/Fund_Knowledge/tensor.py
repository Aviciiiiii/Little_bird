import torch
import numpy as np

data = [[1, 2], [3, 4]]
#.tensor：常规生成tensor张量的方法
x_data = torch.tensor(data)
#.array：常规生成numpy矩阵的方法
np_array = np.array(data)
#numpy矩阵转tensor张量
x_np = torch.from_numpy(np_array)
print(x_np)
print(x_data)
print(np_array)

#按原始矩阵生成一个行、列一致的零矩阵
x_ones = torch.zeros_like(x_data)
print(x_ones)
#按原始矩阵行、列生成一个一样的随机矩阵
x_rand = torch.rand_like(x_data, dtype = torch.float)
print(x_rand)

ten8 = (2, 3, )
#.rand生成一个a*b的随机矩阵
rand_tensor = torch.rand(ten8)
#.ones生成一个a*b的数据为1的矩阵
ones_tensor = torch.ones(ten8)
#.zeros生成一个a*b的零矩阵
zeros_tensor = torch.zeros(ten8)
print(rand_tensor)
print(ones_tensor)
print(zeros_tensor)
print(ten8)

tensor = torch.rand(3,4)
print(tensor)
#tensor张量包含参数有：Size(行、列）、数据类型、数据存储设备
print(tensor.shape)
print(tensor.dtype)
print(tensor.device)

ten11 = torch.ones(4, 4)
#第二列的所有数设为0
ten11[:, 1] = 0
print(ten11)

np0 = np.random.random((3, 3))
#第二、三行的第一列设为0
np0[1:3, 1] = 0
print(np0)

#.cat：连接框里的所有矩阵
ten12 = torch.cat([ten11,ten11,ten11])
print(ten12)
print(x_data)
#.T：矩阵转置
print(x_data.T)
#.mul：矩阵里每个元素相对应做乘法
print(x_data.mul(x_data.T))
#.matmul：矩阵乘法
print(x_data.matmul(x_data))
#@：矩阵乘法
print(x_data @ x_data)

#每个元素加5
x_data.add_(5)
print(x_data)

#switch tensor to numpy#
ten17 = torch.ones((2,2))
num17 = ten17.numpy()
print(num17)
ten17.add_(1)
print(ten17)
print(num17)

#switch numpy to tensor#
num18 = np.zeros((2,2))
ten18 = torch.from_numpy(num18)
np.add(num18,1,out = num18)#method of nparray_add
print(num18)
print(ten18)

a = torch.rand(2,2,2,2)
print(a)
