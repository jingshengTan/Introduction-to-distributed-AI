"""
ftp 文件服务
多线程并发和套接字练习
"""
##########  导入库  ############
from socket import *
from threading import Thread
import sys,os
import time
import pathlib
import random
import codecs
from PIL import Image
import numpy as np
from tqdm import tqdm
from time import sleep
import config
import json
##########  全局变量  ############

dataSeverIP_Port = config.dataSeverIP_Port
Data_root_dir="./dataset/cifar10/train"
PathListDir = "./PathListDir"   #代表图片地址目录存放地
workNUM = config.workerNUM  #工作节点数
DataParallelMode = config.DataParallelMode  #数据并行模式：随机采样或者置乱切分
# 获取数据库数据下载目录
class FTPList():
    def __init__(self,Data_root_dir,PathListDir,workNUM,DataParallelMode):
        self.PathListDir = PathListDir
        if not os.path.exists(self.PathListDir):
            os.mkdir(self.PathListDir)
        self.Data_root_dir = Data_root_dir
        self.workNUM=workNUM
        self.DataParallelMode=DataParallelMode
    def getImagePath_Label(self):
        data_root = pathlib.Path(self.Data_root_dir)
        all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
        ##将所有图片地址打乱
        random.shuffle(all_image_path)
        # get labels' names
        label_names = sorted(item.name for item in data_root.glob('*/'))
        # dict: {label : index}
        label_to_index = dict((label, index) for index, label in enumerate(label_names))
        # get all images' labels
        #all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in
         #                  all_image_path]

        return all_image_path, label_to_index

    def CreateDataPathList(self):
        print("CreateDataPathList")
        all_image_path,label_to_index=self.getImagePath_Label()
        for i in range(self.workNUM):

            dir = self.PathListDir + "/" + str(i) + ".txt"

            imageNum_per_work = len(all_image_path) // self.workNUM
            ##如果数据并行模式为随机采样
            if self.DataParallelMode == "RandomSampling":
                numpool = [i for i in range(len(all_image_path))]
                imgForWorkerIndex = random.sample(numpool, imageNum_per_work)
                with codecs.open(dir, mode='w', encoding='utf-8') as file_txt:
                    for j in range(imageNum_per_work):
                        file_txt.write(all_image_path[imgForWorkerIndex[j]] + '\n')
            ##如果数据并行模式为置乱切分
            elif self.DataParallelMode == "ShuffleCutting":
                with codecs.open(dir, mode='w', encoding='utf-8') as file_txt:
                    for index in range(imageNum_per_work):
                        #print(all_image_path[i*imageNum_per_work+index])
                        file_txt.write(all_image_path[i*imageNum_per_work+index]+'\n')
        return label_to_index
##打印进度条函数
def process_bar(percent, start_str='', end_str='', total_length=0):
    bar = ''.join(["\033[31m%s\033[0m"%'   '] * int(percent * total_length)) + ''
    bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent*100) + end_str
    print(bar, end='', flush=True)
# 处理客户端各种请求
class FTPServer(Thread):
    def __init__(self,connfd,PathListDir,Data_root_dir,label_to_index):
        super().__init__()
        self.connfd = connfd
        self.PathListDir=PathListDir
        self.Data_root_dir=Data_root_dir
        self.label_to_index=label_to_index
    ##向工作节点发送标签与index的关系字典
    def send_label_to_index(self):
        data=json.dumps(self.label_to_index)
        self.connfd.send(data.encode())
    def send_list(self,file_name):
        # 判断文件库是否为空
        file_dir = self.PathListDir + '/' + str(file_name)
        print("发送的文件地址为 = ="+file_dir)
        if not os.path.exists(file_dir):
            self.connfd.send("NotExist".encode())
            print("文件"+file_name+"不存在")
            return
        else:
            with codecs.open(file_dir,mode='r',encoding='utf-8') as file:
                allData=file.read()
            self.connfd.send(str(len(allData)).encode())
            time.sleep(0.1)
            self.connfd.send(allData.encode())
    def send_MakeDir(self):
        Data_root_dir = pathlib.Path(self.Data_root_dir)
        label_names = sorted(item for item in Data_root_dir.glob('*/'))
        dir = ''
        for i in range(len(label_names)):
            dir = dir+str(label_names[i])+"#"
        #print(dir)
        self.connfd.send(dir.encode())
        # dirNum = len(label_names)
        # print(dirNum)
        # self.connfd.send(str(dirNum).encode())
        #
        # for i in range(dirNum):
        #     self.connfd.send(str(label_names[i]).encode())
        #     sleep(0.2)
    # 下载功能 给客户端发文件
    def UseSocketSendData(self,DataNum):
        print("DataNum="+DataNum)
        print("开始传输图片：")
        for i in range(int(DataNum)):
            fileName = self.connfd.recv(1024).decode()
            print(fileName)
            img = Image.open(fileName)
            img = np.array(img) ##将图片转化为numpy.ndarray格式
            print(img)
            stringImg = img.tostring()
            ##由于所有图片大小均为3072，一次最多可接收8k的内容，因此不分批传输
            self.connfd.send(stringImg)
            process_bar(i / int(DataNum) , start_str='',end_str='100%',total_length=15)
            # try:
            #     img = Image.open(fileName)
            #     img = np.array(img) ##将图片转化为numpy.ndarray格式
            #     stringImg = img.tostring()
            #     ##由于所有图片大小均为3072，一次最多可接收8k的内容，因此不分批传输
            #     self.connfd.send(stringImg)
            # except:
            #     # 文件不存在
            #     ack = "文件"+fileName+"不存在"
            #     print(ack)
            #     self.connfd.send("fileNotExist".encode())
    ##断开与相应客户端的连接
    def quitSever(self):
        self.connfd.close()
        print("一工作节点退出连接")

    def run(self):
        while True:
            data = self.connfd.recv(1024).decode() # 接收客户端请求
            # 根据请求类型分情况讨论
            if not data or data == 'E':
                return # 函数结束即线程退出
            elif data == "getDatalist":
                self.connfd.send("R1".encode())
                aimTXT = self.connfd.recv(1024).decode()
                self.send_list(aimTXT)
            elif data == "makedir":
                self.connfd.send("R3".encode())
                self.send_MakeDir()
            elif data == "loaddata":
                self.connfd.send("R2".encode())
                ##接收需要下载的图片数量
                DataNum=self.connfd.recv(128).decode()
                self.UseSocketSendData(DataNum)
            elif data == "redistribution":
                self.connfd.send("R4".encode())
                ftplist = FTPList(Data_root_dir, PathListDir, workNUM, DataParallelMode)
                ftplist.CreateDataPathList()
            elif data == "labelToindex":
                self.connfd.send("R5".encode())
                self.send_label_to_index()
            elif data =='quit':
                self.quitSever()
                break



# 网络并发结构搭建
def main(label_to_index=None):
    # 创建套接字
    sock = socket()
    sock.bind(dataSeverIP_Port)
    sock.listen(3)

    print("Listen the port 8888")
    # 循环链接客户端
    while True:
        try:
            connfd,addr = sock.accept()
            print("客户端地址:",addr)
        except:
            sys.exit("服务退出")

        # 创建新的线程,处理客户端请求 (通过自定义线程类)
        t = FTPServer(connfd,PathListDir,Data_root_dir,label_to_index)
        t.setDaemon(True)  # 主服务退出,其他服务也随之退出
        t.start()  # 运行run



if __name__ == '__main__':
    ftplist=FTPList(Data_root_dir,PathListDir,workNUM,DataParallelMode)
    label_to_index=ftplist.CreateDataPathList()
    main(label_to_index)