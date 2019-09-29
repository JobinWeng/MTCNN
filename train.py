import torch
import torch.nn as nn
import torch.optim as optim
from dataHandle import dataset
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class trainer:
    def __init__(self,net,savePath,datasetPath):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:",self.device)
        self.net = net.to(self.device)
        self.savePath = savePath
        self.datasetPath = datasetPath

        self.conLossFn = nn.BCELoss()
        self.offsetLossFn = nn.MSELoss()

        self.optimizer = optim.Adam(self.net.parameters())

        self.writer = SummaryWriter()

        #继续训练
        if os.path.exists(savePath):
            self.net = torch.load(savePath, map_location=self.device)

    def train(self):
        datas = dataset(self.datasetPath)
        dataLoader = DataLoader(datas,shuffle=True,batch_size=128,num_workers=2)

        epoch = 0
        while True:
            for i, (imgData, conLabel, offsetLabel, landmarkLabel) in enumerate(dataLoader):
                imgData = imgData.to(self.device)
                conLabel = conLabel.to(self.device)
                offsetLabel = offsetLabel.to(self.device)
                landmarkLabel = landmarkLabel.to(self.device)

                outPutCon, outOffset, outlandmark  = self.net(imgData)

                #P 网络进行形状变换
                outPutCon = outPutCon.view(-1,1)
                outOffset = outOffset.view(-1,4)
                outlandmark = outlandmark.view(-1,10)


                #计算置信度损失（剔除部分样本）
                mask = conLabel[:,0]<2   #找出置信度小于2的掩码
                conLabel_ = conLabel[mask]
                outPutCon_ = outPutCon[mask]
                conLoss = self.conLossFn(outPutCon_,conLabel_)

                #偏移量损失(剔除负样本)
                mask = conLabel[:,0]>0   #找出置信度大于0的掩码
                offsetLabel_ = offsetLabel[mask]
                outOffset_ = outOffset[mask]
                offsetLoss = self.offsetLossFn(outOffset_,offsetLabel_)

                #关键点损失
                landmarkLabel_ = landmarkLabel[mask]
                outlandmark_ = outlandmark[mask]
                landmarkLoss = self.offsetLossFn(outlandmark_,landmarkLabel_)

                loss = 0.7*conLoss + 0.2*offsetLoss + 0.1*landmarkLoss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i %50 == 0:

                    loss = loss.cpu().data.item()
                    conLoss = conLoss.cpu().data.item()
                    offsetLoss = offsetLoss.cpu().data.item()
                    landmarkLoss = landmarkLoss.cpu().data.item()

                    print("epoch", epoch, "loss:",loss,"conLoss:",conLoss,"offsetLoss:",offsetLoss,"landmarkLoss",landmarkLoss)
                    self.writer.add_scalars("loss", {
                        "conLoss": conLoss,
                        "offsetLoss": offsetLoss,
                        "landmarkLoss": landmarkLoss,
                        # "sumLoss":loss,
                    }, epoch)

            epoch +=1
            torch.save(self.net, self.savePath)
            if epoch>15:
                break



