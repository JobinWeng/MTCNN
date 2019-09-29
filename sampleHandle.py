import os
from PIL import Image,ImageDraw
import numpy as np
from utils import iou
import traceback

rigionSampleText = r"F:\calebaA\anno\list_bbox_celeba.txt"
pointSampelText = r"F:\calebaA\anno\list_landmarks_celeba.txt"

rigionImg = r"F:\calebaA\img_celeba"
savePath = r"F:\mtcc_Sampel_5"

def sampleHandle(Mode=1, positveCount = 0, negativeCount = 50000, partCount = 0):
    for faceSize in [48]:
        print("creat : ",faceSize)
        #图片存放路径
        positiveImageDir = os.path.join(savePath,str(faceSize),"positve")
        negativeImageDir = os.path.join(savePath, str(faceSize),"negative")
        partImageDir = os.path.join(savePath, str(faceSize), "part")

        for dirPath in [positiveImageDir,negativeImageDir,partImageDir]:
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)
        #打开labe文本
        if Mode == 2:
            negativeSampelText = os.path.join(savePath, str(faceSize), "negative.txt")
            negativeFile = open(negativeSampelText, 'a')
        else:
            positiveSampelText = os.path.join(savePath,str(faceSize),"positive.txt")
            positiveFile = open(positiveSampelText, 'a')
            partSampelText = os.path.join(savePath, str(faceSize), "part.txt")
            partFile = open(partSampelText, 'a')

        maxPartNum = 300000
        maxPositive = 400000
        maxNegative = 500000
        while (positveCount < maxPositive) or (partCount < maxPartNum) or (negativeCount < maxNegative):
            orignText = open(rigionSampleText)
            poinText = open(pointSampelText)
            poinText = poinText.readlines()
            for i, line in enumerate(orignText):
                #过滤第一行
                if i < 2:
                    continue
                print("positveCount:",positveCount)
                print("partCount:", partCount)
                print("negativeCount:", negativeCount)

                poinStrs = poinText[i]

                strs = line.strip().split()
                poinStr = poinStrs.strip().split()

                imgPath = os.path.join(rigionImg,strs[0])
                img = Image.open(imgPath)

                img_w,img_h = img.size
                x1 = float(strs[1])
                y1 = float(strs[2])
                w = float(strs[3])
                h = float(strs[4])
                x2 = x1 + w
                y2 = y1 + h

                #过滤一下不好的样本
                if max(w,h)<40 :
                    continue

                #计算人脸的中心点，为了造出多样本
                cx = x1 + w/2
                cy = y1 + h/2

                #正样本与部分样本
                if Mode != 2:
                    w_ = np.random.randint(int(-w * 0.5), int(w * 0.5))
                    h_ = np.random.randint(int(-h * 0.5), int(h * 0.5))
                    sideLen = np.random.randint(int(min(w, h) * 0.8), np.ceil(max(w, h) * 1.2))
                    cx_ = cx + w_
                    cy_ = cy + h_

                    x1_ = np.maximum((cx_-sideLen/2),0)
                    y1_ = np.maximum((cy_-sideLen/2),0)
                    x2_ = x1_ + sideLen
                    y2_ = y1_ + sideLen

                    #超框处理
                    if x2_ > img_w and y2_ > img_h:
                        w = img_w - x1_
                        h = img_h - y1_
                        sideLen = min(w,h)
                    elif x2_ > img_w:
                        w = img_w - x1_
                        sideLen= w
                    elif y2_ > img_h:
                        h = img_h - y1_
                        sideLen= h

                    # 超框时，重新定义坐标
                    x2_ = x1_ + sideLen
                    y2_ = y1_ + sideLen

                    # 过小的框也不要
                    if sideLen < 24:
                        print("sodeLenOver")
                        continue

                    offset_x1 = (x1 - x1_)/ sideLen
                    offset_y1 = (y1 - y1_)/ sideLen
                    offset_x2 = (x2 - x2_)/ sideLen
                    offset_y2 = (y2 - y2_)/ sideLen

                    offset_x3 = (int(poinStr[1]) - x1_)/ sideLen
                    offset_y3 = (int(poinStr[2]) - y1_)/ sideLen

                    offset_x4 = (int(poinStr[3]) - x2_)/ sideLen
                    offset_y4 = (int(poinStr[4]) - y1_)/ sideLen

                    offset_x5 = (int(poinStr[5]) - x1_)/ sideLen
                    offset_y5 = (int(poinStr[6]) - y1_)/ sideLen

                    offset_x6 = (int(poinStr[7]) - x1_)/ sideLen
                    offset_y6 = (int(poinStr[8]) - y1_)/ sideLen
                    offset_x7 = (int(poinStr[9]) - x2_)/ sideLen
                    offset_y7 = (int(poinStr[10]) - y2_)/ sideLen


                    #截下图的框与图像框坐标，准备作IOU
                    cropBox = np.array([x1_, y1_, x2_, y2_])
                    boxes = np.array([[x1, y1, x2, y2]])

                    iouData =iou(cropBox, boxes)[0]
                    if iouData > 0.65:
                        if positveCount < maxPositive:
                            # 截下图片
                            facCrop = img.crop(cropBox)
                            facResize = facCrop.resize((faceSize, faceSize))

                            positiveFile.write("{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                positveCount, 1, offset_x1, offset_y1, offset_x2, offset_y2, offset_x3, offset_y3,
                                offset_x4, offset_y4, offset_x5, offset_y5, offset_x6, offset_y6, offset_x7, offset_y7))
                            positiveFile.flush()
                            facResize.save(os.path.join(positiveImageDir, "{}.jpg".format(positveCount)))
                            positveCount += 1
                    elif iouData > 0.3 and iouData < 0.45:
                        if partCount < maxPartNum:
                            # 截下图片
                            facCrop = img.crop(cropBox)
                            facResize = facCrop.resize((faceSize, faceSize))

                            partFile.write("{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                partCount, 2, offset_x1, offset_y1, offset_x2, offset_y2, offset_x3, offset_y3,
                                offset_x4, offset_y4, offset_x5, offset_y5, offset_x6, offset_y6, offset_x7, offset_y7))
                            partFile.flush()
                            facResize.save(os.path.join(partImageDir, "{}.jpg".format(partCount)))
                            partCount += 1

                    if (positveCount >= maxPositive) and (partCount >= maxPartNum):
                        break

                    #查看信息
                    # print("iou:",iouData)
                    # print("cropBox:",cropBox)
                    # print("boxes:",boxes[0])
                    # print("sideLen:",sideLen)
                    #查看样本信息
                    # draw = ImageDraw.Draw(img)
                    # draw.rectangle(((x1,y1),(x2,y2)),outline='red',width=1)
                    # draw.rectangle(((x1_, y1_), (x2_, y2_)), outline='green', width=1)
                    # img.show()

                #造负样本
                elif Mode == 2:
                    sideLen = np.random.randint(0.8*(np.minimum(w,h)),1.2*(np.maximum(w,h)))
                    x1_ = np.random.randint(0,0.6*img_w)
                    y1_ = np.random.randint(0,0.6*img_h)
                    x2_ = x1_+ sideLen
                    y2_ = y1_+ sideLen

                    if (x1_+ sideLen) > img_w or (y1_+ sideLen) > img_h:
                        continue

                    box = np.array([x1_,y1_,x2_,y2_])
                    boxes = np.array([[x1, y1, x2, y2]])
                    if iou(box,boxes) == 0:
                        facCrop = img.crop(box)
                        facResize = facCrop.resize((faceSize, faceSize))
                        facResize.save(os.path.join(negativeImageDir, "{}.jpg".format(negativeCount)))
                        negativeFile.write("{0}.jpg 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(negativeCount))
                        negativeFile.flush()
                        negativeCount += 1

                    if negativeCount > maxNegative:
                        break



        if Mode != 2:
            partFile.close()
            positiveFile.close()
        else:
            negativeFile.close()






if __name__ == '__main__':
    try:
        sampleHandle(Mode=1 )
        # negativeFile = open(r'F:\mtcc_Sampel_5\48\negative.txt', 'a')
        #
        # for i in range(500000):
        #     negativeFile.write("{0}.jpg 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(i))
        #     negativeFile.flush()


    except Exception as e:
        traceback.print_exc()