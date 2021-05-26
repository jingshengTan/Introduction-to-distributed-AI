import os
import threading
from datetime import datetime

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
from torch import optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import PicLoader
import argparse
import config
from time import *
from torchvision.models.resnet import ResNet, Bottleneck
##全局变量

EPOCH = config.epoch
BATCH_SIZE = config.batch_size_per_machine
num_classes = config.num_classes
traindir = config.traindir
testdir = config.testdir
num_batches_to_show_loss = config.num_batches_to_show_loss

tongbuDatadir = config.tongbuDatadir
##
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# -------------------------
def timed_log(text):
    print(f"{datetime.now().strftime('%H:%M:%S')} {text}")

##
##定义参数服务器
# ---功能如下---
# 更新参数、发送模型
class BatchUpdateParameterServer(object):

    def __init__(self):
        self.lunshu_ps = 0
        self.model = torchvision.models.resnet50(num_classes=num_classes)
        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)

    def get_model(self):
        return self.model

    def get_lunshu(self):
        return self.lunshu_ps

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads, name, lunshu_worker):
        ##获取参数服务器此时的值
        self = ps_rref.local_value()
        with self.lock:
            ##将梯度加到参数服务模型上
            chengfa = self.lunshu_ps - lunshu_worker
            fut = self.future_model
            for p, g in zip(self.model.parameters(), grads):
                p.grad += g / (chengfa + 1)

            self.optimizer.step()
            self.optimizer.zero_grad()
            timed_log(f"PS used {name}'s grad,参数服务器={self.lunshu_ps} 轮,{name}={lunshu_worker} 轮")
            self.lunshu_ps = self.lunshu_ps + 1
            fut.set_result((self.model, self.lunshu_ps))

            self.future_model = torch.futures.Future()

            return fut


class Trainer(object):

    def __init__(self, ps_rref, device):
        self.ps_rref = ps_rref
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device

    def train(self):
        name = rpc.get_worker_info().name
        # 获取加入时的参数服务器初始化模型
        #工人1使用cuda1
        if name =='trainer1':
            m = self.ps_rref.rpc_sync().get_model().cuda(self.device[2])

            lunshu_worker = self.ps_rref.rpc_sync().get_lunshu()

            # 加载数据集开始训练
            res = []
            traindir = tongbuDatadir + name
            trainset = PicLoader.MyPicDataset(traindir)
            trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                                     shuffle=True, num_workers=0)
            for epoch in range(EPOCH):
                running_loss = 0.0
                T1 = time()
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    inputs = inputs.to(self.device[2])
                    labels = labels.to(self.device[2])
                    # timed_log(f"{name} at epoch={epoch} processing one batch")
                    loss = self.loss_fn(m(inputs), labels)
                    loss.backward()
                    ##输出每个batch的平均损失
                    running_loss += loss.item()
                    if i % num_batches_to_show_loss == num_batches_to_show_loss - 1:  # print every 10 mini-batches

                        processTime = time() - T1  ##每num_batches_to_show_loss个bath的运行时间
                        T1 = time()
                        print('processTime: %.3f,[%d, %5d] loss: %.3f' %
                              (processTime, epoch + 1, i + 1, running_loss / num_batches_to_show_loss))
                        ##保存的是num_batches_to_show_loss个batch的执行总时间，以及平均loss
                        res.append([processTime, running_loss / num_batches_to_show_loss])
                        running_loss = 0.0

                    # timed_log(f"{name} at epoch={epoch} reporting grads")
                    # 传输梯度到参数服务器
                    m, lunshu_worker = rpc.rpc_sync(
                        self.ps_rref.owner(),
                        BatchUpdateParameterServer.update_and_fetch_model,
                        args=(self.ps_rref, [p.grad for p in m.cpu().parameters()], name, lunshu_worker),
                    )
                    ##加载模型
                    ##不同的工作节点使用不同的设备
                    m = m.cuda(self.device[2])
            ##工人1使用cpu，cuda0
        elif name == 'trainer2':
            m = self.ps_rref.rpc_sync().get_model()
            lunshu_worker = self.ps_rref.rpc_sync().get_lunshu()
            # 加载数据集开始训练
            res = []
            traindir = tongbuDatadir + name
            trainset = PicLoader.MyPicDataset(traindir)
            trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                                     shuffle=True, num_workers=0)
            for epoch in range(EPOCH):
                running_loss = 0.0
                T1 = time()
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    inputs = inputs.to(self.device[0])
                    labels = labels.to(self.device[1])
                    
                    # timed_log(f"{name} at epoch={epoch} processing one batch")
                    loss = self.loss_fn(m(inputs).to(self.device[1]), labels)

                   
                    loss.backward()
                    ##输出每个batch的平均损失
                    running_loss += loss.item()
                    if i % num_batches_to_show_loss == num_batches_to_show_loss - 1:  # print every 10 mini-batches

                        processTime = time() - T1  ##每num_batches_to_show_loss个bath的运行时间
                        T1 = time()
                        print('processTime: %.3f,[%d, %5d] loss: %.3f' %
                              (processTime, epoch + 1, i + 1, running_loss / num_batches_to_show_loss))
                        ##保存的是num_batches_to_show_loss个batch的执行总时间，以及平均loss
                        res.append([processTime, running_loss / num_batches_to_show_loss])
                        running_loss = 0.0

                    # timed_log(f"{name} at epoch={epoch} reporting grads")
                    # 传输梯度到参数服务器
                    m, lunshu_worker = rpc.rpc_sync(
                        self.ps_rref.owner(),
                        BatchUpdateParameterServer.update_and_fetch_model,
                        args=(self.ps_rref, [p.grad for p in m.cpu().parameters()], name, lunshu_worker),
                    )
                    ##加载模型

                    ##不同的工作节点使用不同的设备
                    ##resnet50一共参数向量一共有638个
                    jishu = 0
                    for p in m.parameters():
                        if jishu >=319:
                            p.data=p.data.cuda(self.device[1])
                        jishu = jishu+1
        PicLoader.SaveResult('./jiqun', name, res)
        print('Finished Training')


##运行训练者
def run_trainer(ps_rref, device):
    trainer = Trainer(ps_rref, device)
    trainer.train()


##运行参数服务器
def run_ps(world_size):
    timed_log("Start training")
    ps_rref = rpc.RRef(BatchUpdateParameterServer())
    futs = []
    for r in range(1, world_size):
        futs.append(
            rpc.rpc_async(f'trainer{r}', run_trainer, args=(ps_rref, ['cpu','cuda:0','cuda:1'],))
        )
    ##等待所有的工作节点完成训练
    torch.futures.wait_all(futs)
    timed_log("Finish training")


def run(rank, world_size, rpc_backend_options):
    if rank != 0:
        rpc.init_rpc(
            f"trainer{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
        # trainer passively waiting for ps to kick off training iterations
    else:
        rpc.init_rpc(
            "ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
        run_ps(world_size)

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synchronous Parameter-Server RPC based training")
    parser.add_argument(
        "--world_size",
        type=int,
        default=3,
        help="""Total number of participating processes. Should be the sum of
            master node and all training nodes.""")
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Global rank of this process. Pass in 0 for master.")

    parser.add_argument(
        "--master_addr",
        type=str,
        default="10.128.217.80",
        help="""Address of master, will default to 10.128.217.80 if not provided.
            Master must be able to accept network traffic on the address + port.""")
    parser.add_argument(
        "--master_port",
        type=str,
        default="29500",
        help="""Port that master is listening on, will default to 29500 if not
            provided. Master must be able to accept network traffic on the host and port.""")

    args = parser.parse_args()
    assert args.rank is not None, "must provide rank argument."
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ["MASTER_PORT"] = '29500'
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=0  # infinite timeout
    )
    rank = args.rank
    world_size = args.world_size
    run(rank, world_size, options)