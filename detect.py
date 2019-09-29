import torch
from PIL import Image
from PIL import ImageDraw,ImageFont
import numpy as np
import utils
from torchvision import transforms
import traceback
import time
import gc

class Detector:
    def __init__(self,pNetPath='./model/Pnet.pt', rNetPath='./model/Rnet.pt' ,oNetPath='./model/Onet.pt'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pNet = torch.load(pNetPath, map_location=self.device)
        self.rNet = torch.load(rNetPath, map_location=self.device)
        self.oNet = torch.load(oNetPath, map_location=self.device)

        self.pScale = 0.75
        self.pCon = 0.75
        self.rCon = 0.85
        self.oCon = 0.999
        self.pNms = 0.5
        self.rNms = 0.3
        self.oNms = 0.7

        self.pNet.eval()
        self.rNet.eval()
        self.oNet.eval()

        self.imgTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(std=[0.5], mean=[0.5])
        ])

    def detect(self,image):
        t1 = time.time()
        pBoxs = self.pNetDetect(image)
        print("P_TIME:",time.time()-t1)
        if len(pBoxs)== 0:
            print("P_Detect Not box")

        t1 = time.time()
        rBoxs = self.rNetDetect(image,pBoxs)
        print("R_TIME:", time.time() - t1)
        if len(rBoxs) == 0:
            print("R_Dect Not box")

        t1 = time.time()
        oBoxs = self.oNetDetect(image,rBoxs)
        print("O_TIME:", time.time() - t1)
        if len(oBoxs) == 0:
            print("O_Dect Not box")

        w = (oBoxs[:,2] - oBoxs[:,0])
        h = (oBoxs[:,3] - oBoxs[:,1])
        x_c = w / 2 + oBoxs[:,0]
        y_c = h / 2 + oBoxs[:,1]
        oBoxs[:,0] = x_c-0.4*w
        oBoxs[:,1] = y_c - 0.325 * h
        oBoxs[:,2] = x_c+0.4*w
        oBoxs[:,3] = y_c + 0.325 * h

        return oBoxs.cpu().data.numpy()

    def pNetDetect(self,imge):
        boxes = []
        w, h = imge.size

        minSideLen = min(w,h)
        scale = 1

        while minSideLen >12:
            imgData = self.imgTransform(imge)
            imgData = imgData.unsqueeze(0)
            imgData = imgData.to(self.device)

            cons, offsets,_ = self.pNet(imgData)
            idxs = torch.nonzero(torch.gt(cons[0][0],self.pCon))
            boxes.extend(self.returnBox(idxs, offsets[0],cons[0][0], scale))

            scale *= self.pScale

            _w = int(w*scale)
            _h = int(h*scale)

            imge = imge.resize((_w,_h))
            minSideLen = min(_w,_h)

            del imgData,cons,offsets,idxs,_
            gc.collect()

        boxes = torch.stack(boxes)
        return utils.nms(boxes,self.pNms)

    def rNetDetect(self, imge,pNetBoxes):

        #无框返回
        if len(pNetBoxes) == 0:
            return []

        imgDateSets = []
        pNetBoxes = utils.convert_to_square(pNetBoxes)

        #扣下P网络的图
        for pBox in pNetBoxes:

            imgeCrop = imge.crop((int(pBox[0]),int(pBox[1]),int(pBox[2]),int(pBox[3])))
            imgeCrop = imgeCrop.resize((24,24))

            imgData = self.imgTransform(imgeCrop)
            imgDateSets.append(imgData)

        #转换图片
        imgDateSets = torch.stack(imgDateSets)
        imgDateSets = imgDateSets.to(self.device)

        cons, offsets, landMark = self.rNet(imgDateSets)

        return self.boxDetect(pNetBoxes,cons,offsets,conMax=self.rCon,nmsMax=self.rNms,landMark=landMark)

    def oNetDetect(self,imge,rNetBoxes):
        imgDataset = []

        #无框返回
        if len(rNetBoxes) == 0:
            return np.array([])

        rNetBoxes = utils.convert_to_square(rNetBoxes)

        for box in rNetBoxes:
            imgeCRop = imge.crop((int(box[0]),int(box[1]),int(box[2]),int(box[3])))
            imgeCrop = imgeCRop.resize((48,48))

            imgData = self.imgTransform(imgeCrop)
            imgDataset.append(imgData)

        imgDataset = torch.stack(imgDataset)
        imgDataset = imgDataset.to(self.device)
        cons,offsets, landmak = self.oNet(imgDataset)

        return self.boxDetect(rNetBoxes, cons, offsets, conMax=self.oCon, nmsMax=self.oNms, iouMode='min',landMark=landmak)

    #r网络，o网络检测
    def boxDetect(self, inputBoxes, cons, offsets, landMark, conMax, nmsMax=0.3, iouMode='inter'):
        mask = cons > conMax
        mask_index = mask.nonzero()[:,0]

        cons = cons[mask]
        offsets = torch.index_select(offsets,dim=0,index=mask_index)
        boxes = torch.index_select(inputBoxes,dim=0,index=mask_index)
        landMark = torch.index_select(landMark,dim=0,index=mask_index)
        if cons.size(0) == 0:
            return []

        #筛选R网络的框
        w = boxes[:,2] - boxes[:,0]
        h = boxes[:,3] - boxes[:,1]

        x1 = boxes[:,0] + w * offsets[:,0]
        y1 = boxes[:,1] + h * offsets[:,1]
        x2 = boxes[:,2] + w * offsets[:,2]
        y2 = boxes[:,3] + h * offsets[:,3]

        # if landMark == None:
        #     return utils.nms(torch.stack([x1, y1, x2, y2, cons], dim=1), thresh=nmsMax, mode=iouMode)
        # else:
        x3 = boxes[:, 0] + w * landMark[:, 0]
        y3 = boxes[:, 1] + h * landMark[:, 1]
        x4 = boxes[:, 2] + w * landMark[:, 2]
        y4 = boxes[:, 1] + h * landMark[:, 3]
        x5 = boxes[:, 0] + w * landMark[:, 4]
        y5 = boxes[:, 1] + h * landMark[:, 5]
        x6 = boxes[:, 0] + w * landMark[:, 6]
        y6 = boxes[:, 1] + h * landMark[:, 7]
        x7 = boxes[:, 2] + w * landMark[:, 8]
        y7 = boxes[:, 3] + h * landMark[:, 9]

        return utils.nms(torch.stack([x1, y1, x2, y2, cons, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7], dim=1), thresh=nmsMax, mode=iouMode)

    #P网络返回原图
    def returnBox(self, startIndex, offset, con, scale, stride=2, sideLen=12):

        _x1 = (startIndex[:,1].float() * stride)/scale
        _y1 = (startIndex[:,0].float() * stride)/scale
        _x2 = (startIndex[:,1].float() * stride + sideLen)/scale
        _y2 = (startIndex[:,0].float() * stride + sideLen)/scale

        ow = _x2 - _x1
        oh = _y2 - _y1

        x1 = _x1 + ow * offset[0][startIndex[:,0],startIndex[:,1]]
        y1 = _y1 + oh * offset[1][startIndex[:,0],startIndex[:,1]]
        x2 = _x2 + ow * offset[2][startIndex[:,0],startIndex[:,1]]
        y2 = _y2 + oh * offset[3][startIndex[:,0],startIndex[:,1]]

        return torch.stack([x1, y1, x2, y2, con[startIndex[:,0],startIndex[:,1]]],dim=1)


if __name__ == '__main__':
    imgFile = r"C:\Users\Administrator\Desktop\测试\4560.jpg"
    # imgFile = r"F:\标注完的\img\000007.jpg"
    # imgFile = r"F:\jobin\MTCC\SampleData03\24\positve\5.jpg"
    img = Image.open(imgFile)
    # img_input = img.convert("L")
    try:
        # detctor = Detector(pNetPath=r"model\batchSize=1024\Pnet.pt",rNetPath=r"model\batchSize=1024\Rnet.pt",
        #                    oNetPath=r"model\batchSize=1024\Onet.pt")
        # boxes = detctor.rNetDetect(img, np.array([[0, 0, 24, 24]]))
        # print(boxes)

        detctor = Detector()
        boxes = detctor.detect(img)
        imDraw = ImageDraw.Draw(img)

        for box in boxes:
            position = (int(box[0]),int(box[1]),int(box[2]),int(box[3]))
            imDraw.rectangle(position, outline='red')
            position = (int(box[0]), int(box[1]))
            imDraw.text(position, str(box[4]), spacing=0, align='left')
            i =5
            while i< 15:
                position = (int(box[i]), int(box[i+1]), int(box[i]+2), int(box[i+1]+2))
                i += 2
                imDraw.rectangle(position, outline='red')
        # img.save("./model/测试.jpg")
        img.show()
    except Exception as e:
        traceback.print_exc()