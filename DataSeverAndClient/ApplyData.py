"""
ftp 客户端
c/s模式   发送请求 获取结果
"""

from socket import *
import time
import sys
import config
import cv2
import numpy as np
import time
from PIL import Image
import os
import json
linuxPath = os.getcwd()+'/'
DataDispatcherIP_Port = config.dataSeverIP_Port
worker_index =  config.workerIndex
##进度条显示函数
def process_bar(percent, start_str='', end_str='', total_length=0):
    bar = ''.join(["\033[31m%s\033[0m"%'   '] * int(percent * total_length)) + ''
    bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent*100) + end_str
    print(bar, end='', flush=True)
class ApplyDatalist():
    def __init__(self,worker_index,DataDispatcherIP_Port,sockfd):

        self.worker_index = worker_index
        self.DataDispatcherIP_Port=DataDispatcherIP_Port
        self.sockfd=sockfd
    def getDatalistFormDispathcher(self):
        ##发送命令
        self.sockfd.send("getDatalist".encode())
        ##发送目标txt
        ack = self.sockfd.recv(1024).decode()
        if ack == "R1":
            print("服务器已收到getDatalist指令")
            aimTXT = str(self.worker_index)+'.txt'
            self.sockfd.send(aimTXT.encode())
        else:
            print("！！！错误！！！服务器未收到getDatalist指令")
            return
        dataLen = self.sockfd.recv(1024).decode()
        if dataLen=="NotExist":
            print("服务器不存在文件"+aimTXT)
            return
        else:
            print("预计接收数据量为"+dataLen+"比特")
        recevied_size = 0  # 接收客户端发来数据的计算器
        recevied_data = b''  # 客户端每次发来内容的计数器

        while recevied_size < int(dataLen):  # 当接收的数据大小 小于 客户端发来的数据
            data_res = self.sockfd.recv(4096)
            recevied_size += len(data_res)  # 每次收到的服务端的数据有可能小于1024，所以必须用len判断
            recevied_data += data_res
        else:
            recevied_data = recevied_data.decode()
            RemoteDataList=recevied_data.split('\n')
            RemoteDataList.pop(-1)
            return RemoteDataList
    def makedir(self):
        self.sockfd.send("makedir".encode())
        ack = self.sockfd.recv(128).decode()
        if ack == "R3":
            print("服务器已收到makedir指令")
        else:
            print("error!!!服务器未收到makedir指令")
            return
        dir = self.sockfd.recv(1024)
        dir=dir.decode().split('#')
        dir.pop(-1)

        #dirNum = self.sockfd.recv(128)
        for i in range(len(dir)):
            dddd = '/'.join(dir[i].split('\\'))
            dddd = linuxPath + dddd
            if not os.path.exists(dir[i]):
                os.makedirs(dir[i])
            else:
                continue

    def LoadUseSocket(self,RemoteDataList):
        DataNum= len(RemoteDataList)

        self.sockfd.send("loaddata".encode())
        ack = self.sockfd.recv(1024).decode()
        if ack =="R2":
            print("服务器已收到下载数据指令")
            ##向服务器发送需要下载的图片数量
            self.sockfd.send(str(DataNum).encode())

        else:
            print("error!!服务器未收到下载数据指令")
            return
        startLoadfile = time.time()
        print("开始传输图片")
        for i in range(DataNum):
            self.sockfd.send(RemoteDataList[i].encode())
            #print(RemoteDataList[i])

            #返回的ack可能是图片流，也可能是”图片不存在“
            ack = self.sockfd.recv(4096)
            img = np.frombuffer(ack, np.uint8)
            img = np.reshape(img,[32,32,3])
            img = Image.fromarray(img)
            img.save(RemoteDataList[i])
            process_bar(i / DataNum , start_str='', end_str='100%', total_length=15)
        endLoadfile = time.time()
        seconds = endLoadfile-startLoadfile
        fps = DataNum/ seconds
        print("图片下载结束，fps={}".format(fps))
    def new_shuffle(self):
        self.sockfd.send("redistribution".encode())
        ack = self.sockfd.recv(144).decode()
        if ack == "R4":
            print("服务器已收到重新分配指令")
    def getLabel_to_index(self):
        self.sockfd.send("labelToindex".encode())
        ack = self.sockfd.recv(144).decode()
        if ack == "R5":
            print("服务器已收到获取label_to_index字典指令")
            label_to_index=self.sockfd.recv(1024).decode()
            label_to_index=json.loads(label_to_index)
            return label_to_index
        else:
            print("error!!!R5")

    def quitSever(self):
        self.sockfd.send("quit".encode())
        self.sockfd.close()


# 链接服务端，获取数据，数据并行的方式在服务器被定义，与工作节点无关的方式：
def ApplyData():
    s = socket()
    s.connect(DataDispatcherIP_Port)
    applydata = ApplyDatalist(worker_index, DataDispatcherIP_Port, s)
    ##获取需要下载的图片地址
    RemoteDataList = applydata.getDatalistFormDispathcher()
    ##获取需要创建的目录并创建
    applydata.makedir()
    ##向全程拉取图片
    applydata.LoadUseSocket(RemoteDataList)
    label_to_index = applydata.getLabel_to_index()
    ##结束后关闭连接
    applydata.quitSever()
    return label_to_index
#连接服务器，请求服务器重新分配数据，分配完后仍需重新applydata：
def Redistribution():
    s = socket()
    s.connect(DataDispatcherIP_Port)
    applydata = ApplyDatalist(worker_index, DataDispatcherIP_Port, s)
    ##控制服务器重新分配数据
    RemoteDataList = applydata.new_shuffle()
    ##结束后关闭连接
    applydata.quitSever()
##用于测试各项功能
def test():
    s = socket()
    s.connect(DataDispatcherIP_Port)
    applydata = ApplyDatalist(worker_index, DataDispatcherIP_Port, s)
    applydata.getLabel_to_index()
    applydata.quitSever()
if __name__ == '__main__':
     test()
