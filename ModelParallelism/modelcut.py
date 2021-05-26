import threading
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torchvision.models.resnet import Bottleneck
import config
import torch.distributed.autograd as dist_autograd
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from torch.utils.data import Dataset, DataLoader
import PicLoader
import os
import time
import torch.multiprocessing as mp
from time import *
##全局变量

EPOCH = 1
BATCH_SIZE = config.batch_size_per_machine
num_classes = config.num_classes
trainDatadir = config.traindir

##定义模型
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNetBase(nn.Module):
    def __init__(self, block, inplanes, num_classes=10,
                groups=1, width_per_group=64, norm_layer=None):
        super(ResNetBase, self).__init__()

        self._lock = threading.Lock()
        self._block = block
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = inplanes
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

    def _make_layer(self, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * self._block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * self._block.expansion, stride),
                norm_layer(planes * self._block.expansion),
            )

        layers = []
        layers.append(self._block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * self._block.expansion
        for _ in range(1, blocks):
            layers.append(self._block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.parameters()]

class ResNetShard1(ResNetBase):
    def __init__(self, device, *args, **kwargs):
        super(ResNetShard1, self).__init__(
            Bottleneck, 64, num_classes=num_classes, *args, **kwargs)

        self.device = device
        self.seq = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            self._norm_layer(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(64, 3),
            self._make_layer(128, 4, stride=2)
        ).to(self.device)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out =  self.seq(x)
        return out.cpu()

class ResNetShard2(ResNetBase):
    def __init__(self, device, *args, **kwargs):
        super(ResNetShard2, self).__init__(
            Bottleneck, 512, num_classes=num_classes, *args, **kwargs)

        self.device = device
        self.seq = nn.Sequential(
            self._make_layer(256, 6, stride=2),
            self._make_layer(512, 3, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        ).to(self.device)

        self.fc =  nn.Linear(512 * self._block.expansion, num_classes).to(self.device)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.fc(torch.flatten(self.seq(x), 1))
        return out.cpu()
class DistResNet50(nn.Module):
    def __init__(self, num_split, workers,issplit, *args, **kwargs):
        super(DistResNet50, self).__init__()

        self.num_split = num_split
        self.issplit = issplit
        # Put the first part of the ResNet50 on workers[0]
        self.p1_rref = rpc.remote(
            workers[0],
            ResNetShard1,
            args = ("cuda:0",) + args,
            kwargs = kwargs
        )

        # Put the second part of the ResNet50 on workers[1]
        self.p2_rref = rpc.remote(
            workers[1],
            ResNetShard2,
            args = ("cuda:1",) + args,
            kwargs = kwargs
        )

    def forward(self, xs):
        if not self.issplit:
            t = time()
            x_rref = rpc.RRef(xs)
            y_rref = self.p1_rref.remote().forward(x_rref)
            z_fut = self.p2_rref.rpc_async().forward(y_rref)
            torch.futures.wait_all(z_fut)
            print(f"执行时间{time()-t}")
        else:
            t = time()
            out_futures = []
            for x in iter(xs.split(self.num_split, dim=0)):
                x_rref = rpc.RRef(x)
                y_rref = self.p1_rref.remote().forward(x_rref)
                z_fut = self.p2_rref.rpc_async().forward(y_rref)
                out_futures.append(z_fut)
            print(f"执行时间{time()-t}")
            return torch.cat(torch.futures.wait_all(out_futures))

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
        return remote_params

##定义训练循环

def run_master(num_split,issplit):
    # put the two model parts on worker1 and worker2 respectively
    model = DistResNet50(num_split, ["worker1", "worker2"],issplit)
    loss_fn = nn.CrossEntropyLoss()
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.001
    )

    trainset = PicLoader.MyPicDataset(trainDatadir)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=0)

    ##只执行单个epoch
    for i,data in enumerate(trainloader,0):
        inputs,labels = data

        with dist_autograd.context() as context_id:
            outputs = model(inputs)
            dist_autograd.backward(context_id, [loss_fn(outputs, labels)])
            opt.step(context_id)



def run_worker(rank, world_size, num_split,issplit):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)

    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_master(num_split,issplit)
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        pass

    # block until all rpcs finish
    rpc.shutdown()

if __name__=="__main__":

    world_size = 3
    res = []
    issplit = True
    for num_split in [ 2, 4, 8,16]:
        tik = time()
        mp.spawn(run_worker, args=(world_size, num_split,issplit), nprocs=world_size, join=True)
        tok = time()
        res.append([num_split,tok - tik])
        PicLoader.SaveResult('./modelcut','split',res)
        print(f"number of splits = {num_split}, execution time = {tok - tik}")

    ##不切片的运行时间
    issplit = False
    res = []
    tik = time()
    mp.spawn(run_worker, args=(world_size, 32, issplit), nprocs=world_size, join=True)
    tok = time()
    res.append([num_split, tok - tik])
    PicLoader.SaveResult('./modelcut', 'notsplit', res)
    print(f"number of splits = {num_split}, execution time = {tok - tik}")