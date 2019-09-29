import torch
import os
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms

class dataset(data.Dataset):
    def __init__(self,paths):
        self.paths = paths
        self.dataSet = []
        self.pathIndex = []   #第几个路径

        self.imgTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(std=[0.5],mean=[0.5])
        ])
        for i,path in enumerate(paths) :
            self.dataSet.extend(open(os.path.join(path,"positive.txt")).readlines())
            self.dataSet.extend(open(os.path.join(path, "part.txt")).readlines())
            self.dataSet.extend(open(os.path.join(path, "negative.txt")).readlines())

            #记录批次路径
            "len(self.dataSet),将3种数据集的长度加起来""因为你输入的路径其实不止一个，所以长度应该是不同的"
            "不同路径，对应不同长度，根据下面内容推理出：目标是1~2个不同路径的数据集"
            self.pathIndex.append(len(self.dataSet))

    def __len__(self):
        return len(self.dataSet)

    def __getitem__(self, index):
        strs = self.dataSet[index].strip().split()

        cons = float(strs[1])

        if cons == 0:
            kind = "negative"
        elif cons == 1:
            kind = "positve"
        else:
            kind = "part"
        pathIndex =0
        "第几批次路径，根据下面内容推理出：目标是1~2个不同路径的数据集"
        for  pathLen in self.pathIndex:
            if index < pathLen:
                break
            else:
                pathIndex += 1
        img = Image.open(os.path.join(self.paths[pathIndex], kind, strs[0]))
        img = self.imgTransform(img)

        cons = torch.Tensor([cons])
        offset = torch.Tensor([float(strs[2]),float(strs[3]),float(strs[4]),float(strs[5])])
        landmark = torch.Tensor([float(strs[6]),float(strs[7]),float(strs[8]),float(strs[9]),float(strs[10]),
                                 float(strs[11]),float(strs[12]),float(strs[13]),float(strs[14]),float(strs[15])])

        return img, cons, offset, landmark


if __name__ == '__main__':
    test = dataset([r"F:\jobin\MTCC数据集\SampleData05\48"])
    img,_,_,_ = test[59105]


