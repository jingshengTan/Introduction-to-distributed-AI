import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
import pathlib
import random
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import os
def loadPic(path):
    ##得到所有图片的路径
    data_path = pathlib.Path(path)
    all_image_paths = list(data_path.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    ##所有图片数量
    image_count = len(all_image_paths)
    ##打乱图片路径
    random.shuffle(all_image_paths)
    ##得到图片的分类列表
    label_names = sorted(item.name for item in data_path.glob('*/') if item.is_dir())
    label_and_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [label_and_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]
    return all_image_paths,all_image_labels
class MyPicDataset(Dataset):
    def __init__(self,rootdir):
        self.rootdir = rootdir
        self.all_image_paths,self.all_image_labels = loadPic(self.rootdir)
    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        image = self.all_image_paths[item]
        label = self.all_image_labels[item]
        image = Image.open(image).convert('RGB')
        image = np.array(image).astype(np.float32)
        image=np.transpose(image,(2,0,1))
        image /= 255.0
        return image,label

def SaveResult(rootdir,name,Res):
    ##创建文件夹
    if not os.path.exists(rootdir):
        os.makedirs(rootdir)
    f = open(rootdir+'/'+name+'.txt','a')
    for item in Res:
        for data in item:
            f.write('{} '.format(data))
        f.write('\n')
    f.close()


'''
if __name__=="__main__":
    test = MyPicDataset(r'./data/cifar10/train')
    print(test[0])
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainloader = DataLoader(test, batch_size=5,
                             shuffle=True, num_workers=0)



    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for data,label in trainloader:
        print(data.size())
        break
        
'''