import torch
from net import pNet
import numpy as np
from PIL import Image
from torchvision import transforms
import os

# data = torch.Tensor(np.array(Image.open(r"F:\标注完的\img\000001.jpg")))
# data = data.permute(2,0,1)
# data = data.unsqueeze(0)
#
# net = torch.load(r"F:\jobin\MTCC\Pnet.pt", map_location="cpu")
# con, offset = net(data)
#
# con =con[0]
#
# print(con.shape)





# a = transforms.Compose([
#     transforms.ToTensor(),
# ])
#
# b = Image.open(r"F:\标注完的\img\000001.jpg")
# print(b.size)
# # print(a(b))


# a = np.array([[1,2],[3,4],[5,6],[7,8],[9,10]])
# #
# # b = np.array([1,2,3,4,5])
# #
# # print(a[b>3])
# # c = b>3
# # print(c)

# a = torch.Tensor([[[[1,2],[1,2],[1,2]]],[[[1,2],[1,2],[1,2]]]])
#
# print(a.shape)
# print(a[0][0].shape)


# def  add(a):
#     a += 1
#     print("in:",a)
#
# a = 3
#
# add(a)
#
# print(a)


def sampeChage(resize=12,inputPath=r"F:\jobin\MTCC数据集\SampleData05\48\negative", outPath=r"F:\jobin\MTCC数据集\SampleData05\12\negative"):
    files = os.listdir(inputPath)

    for i, file in enumerate(files):
        img = Image.open(os.path.join(inputPath,file))
        resizeImg = img.resize((resize,resize))
        resizeImg.save(os.path.join(outPath,file))

        if i >250000:
            break

if __name__ == '__main__':
    sampeChage()

