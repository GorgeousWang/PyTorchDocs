
# Author:Elli
import torch
'''
x = torch.ones(2,2,requires_grad=True)
print(x)

y = x + 2
#print(y)
#print(y.grad_fn)

z = y * y * 3
out = z.mean() #求平均值
print(z,out)

a = torch.randn(2,2)
a = ((a * 3)/(a - 1))
#print(a.requires_grad)
a.requires_grad_(True)
#print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

out.backward() #输出里面含有一个标量out.backward()
# -等同于out.backward(torch.tensor(1.))
print(x.grad) #打印梯度 x.grad

'''


x = torch.randn(3,requires_grad=True)
y = x*2
while y.data.norm() < 1000:
    y = y*2
print(y)


v = torch.tensor([0.1, 1.0, 0.0001],dtype=torch.float)
y.backward(v)

print(x.grad) #打印梯度 d(y)/dx

print(x.requires_grad)
print((x ** 2).requires_grad)
#你可以通过将代码包裹在 with torch.no_grad()，
#-来停止对从跟踪历史中 的 .requires_grad=True 的张量自动求导。

with torch.no_grad():
    print((x ** 2).requires_grad)


#1
#2
#3 (1,2,3,5)
#4
#5




