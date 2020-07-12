# Author:Elli
#使得print特性通用于python2.x & Python 3.x
from __future__ import print_function

#导入pytorch
import torch

#1构造一个随机初始化的矩阵
#x = torch.empty(5,3)
#print(x)

#2构造一个矩阵全为0，而且数据类型是long
#x = torch.zeros(5,3,dtype = torch.long)
#print(x)

#3构造 一个张量，直接使用数据5.5和3
#x = torch.tensor([5.5,3])
#print(x)

#4构造 以一个tensor基于已存在的tensor
#x = x.new_ones(5,3,dtype=torch.double)
# 注：new_*  method take in sizes
#print(x)
#x = torch.randn_like(x,dtype=torch.float)
#注：override dtype!
#print(x)

#5获取 张量维度
#print(x.size())
#注意：torch.Size是一个元组（即表中的一行），
# 所以它支持左右的元组操作

#1加法操作
#y = torch.rand(5,3)
#print(x+y)

#2加法操作2 使用add
#print(torch.add(x,y))

#3加法操作3 提供一个输出tensor作为参数
#result= torch.empty(5,3)
#torch.add(x,y,out=result)
#print(result)

#4加法操作4 in-place
#注：adds x to y
#y.add_(x)
#print(y)

#注：任何使张量会发生变化的操作都有一个前缀。
#例如：x.copy(y),x.t_(),将会改变x。
#你可以使用标准的nump类似的索引操作
#print(x[:,1]) #将x转化为行向量

#如果你想改变一个tensor的大小或者形状，你可以使用torch.view
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1,8)
#the size -1 is inferred from other dimensions
#尺寸-1是从其他维度推导出来的
print(x.size(),y.size(),z.size())

#使用.item()来获取一个元素张量tensor中的value
x = torch.randn(1)
print(x)
print(x.item())






















