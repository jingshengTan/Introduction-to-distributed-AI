# 分布式机器学习入门
--------------
本项目使用Pytorch编程，Pytorch版本最好≥1.8，本项目实现了多计算机节点间的分布式机器学习的**数据并行算法与模型并行算法**。具体案例包括如下部分：

 * 如何加载图片数据集
 * 构建数据服务器实现数据切分
 * 数据并行算法的实现，包括：同步SGD，模型平均算法
 * 模型并行算法的实现
 * 混合并行计算
## 如何加载图片数据集
使用PickLoader.py中的MyPicDataset对象，然后用DataLoader加载即可。例子：

	from torch.utils.data import Dataset, DataLoader
	import PicLoader
	trainset = PicLoader.MyPicDataset(traindir)
	trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,shuffle=True, num_workers=0)

## 构建数据服务器实现数据切分
运行DataSeverAndClient中的Datadispatcher.py即为数据服务器
运行运行DataSeverAndClient中的ApplyData.py从数据服务器中获取数据
## 数据并行算法的实现，包括：同步SGD，模型平均算法
详见DataParallelism文件夹。例如，实现同步SGD算法，在三个主机节点分别执行：

	在参数服务器节点：
	   python tongbu.py --rank=0 --master_addr='参数服务器节点ip' --master_port='可用端口'
	工作节点1：
	   python tongbu.py --rank=1 --master_addr='参数服务器节点ip' --master_port='可用端口'
	工作节点2：
	   python tongbu.py --rank=2 --master_addr='参数服务器节点ip' --master_port='可用端口'
## 模型并行算法的实现
同上
## 混合并行计算
同上