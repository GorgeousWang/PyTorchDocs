# Author:Elli

#1加载并归一化CIFAR10，使用torchvision，用它来加载cifar10数据
import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


#2torchvision数据集的输出是范围在[0,1]之间的PILImage
#我们将他们转换成归一化范围为[-1,1]之间的张量Tensors。
transform = transforms.Compose(
    #将多个变换组合在一起
    #第一个是变为张量
    #第二个是标准化
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
#问题：为啥是0.5 0.5 0.5？
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)
#注意：num_workers表示的是进程数字，作者源代码用的是num_workers=2，而多线程代码需要在main函数中运行或者改为单线程
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

# 上面download参数的意义：是否从internet上下载数据集
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

'''
# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
'''

#3 定义一个卷积神经网络





