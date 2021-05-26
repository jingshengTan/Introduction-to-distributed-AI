'''
本代码将使用cifar10数据集、ResNet50网络
在单个GPU上执行
输出结果是精度随时间的变化图
平均每轮的执行时间

'''
import config
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models.resnet import ResNet, Bottleneck
import torchvision.models as models
import PicLoader
from torch.utils.data import DataLoader,Dataset
from time import *
##定义全局变量
EPOCH = 200
BATCH_SIZE = config.batch_size_per_machine
num_classes = config.num_classes
traindir = config.traindir
testdir = config.testdir
num_batches_to_show_loss = config.num_batches_to_show_loss
##加载数据集

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = PicLoader.MyPicDataset(traindir)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0)

testset = PicLoader.MyPicDataset(testdir)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
##---------------

##定义网络


model = models.resnet50(num_classes=num_classes).to('cuda:0')

#-----

##定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
##----

##开始训练

def train(model):
    model.train(True)
    res = []
    st = time()
    for epoch in range(EPOCH):
        running_loss = 0.0
        T1 = time()
        for i, data in enumerate(trainloader, 0):
            #
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.to('cuda:0'))
            loss = criterion(outputs.to('cpu'), labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % num_batches_to_show_loss == num_batches_to_show_loss-1:    # print every 10 mini-batches
                processTime = time() - T1##每num_batches_to_show_loss个bath的运行时间
                T1 = time()
                print('processTime: %.3f,[%d, %5d] loss: %.3f' %
                      (processTime,epoch + 1, i + 1, running_loss / num_batches_to_show_loss))
                ##保存的是num_batches_to_show_loss个batch的执行总时间，以及平均loss
                res.append([processTime,running_loss/num_batches_to_show_loss])
                running_loss = 0.0
    PicLoader.SaveResult('./SingelWork','EOPCH200',res)
    print('Finished Training')
    et = time()
    print(f"单个epoch执行时间为{et-st}")

if __name__ == "__main__":
    train(model)



