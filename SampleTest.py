import os
from PIL import Image ,ImageDraw


# labFile = open(r"F:\标注完的\lable.txt")
#
# labdata = labFile.readlines()
#
# for i, strs in enumerate(labdata):
#     if i == 0:
#         continue
#     data = strs.strip().split(',')
#     img = Image.open(os.path.join("F:\标注完的\img","{}.jpg".format(data[0])))
#     imgDraw = ImageDraw.Draw(img)
#     imgDraw.rectangle((int(data[1]), int(data[2]), int(data[1])+int(data[3]), int(data[2])+int(data[4])),outline="red")
#     img.save(os.path.join("F:\标注完的\label","{}.jpg".format(data[0])))


labFile = open(r"F:\anno\list_bbox_celeba.txt")
keyFile = open(r"F:\anno\list_landmarks_celeba.txt")

labdata = labFile.readlines()
key_data = keyFile.readlines()
for i, strs in enumerate(key_data):
    if i < 2:
            continue
    print('strs',strs)
    data = strs.strip().split()
    print("data:",data)
    img = Image.open(os.path.join(r"F:\anno","{}".format(data[0])))
    imgDraw = ImageDraw.Draw(img)
    imgDraw.rectangle((int(data[1]), int(data[2]), int(data[1])+10, int(data[2])+10),outline="red")
    imgDraw.rectangle((int(data[3]), int(data[4]), int(data[3]) + 10, int(data[4]) + 10), outline="red")
    imgDraw.rectangle((int(data[5]), int(data[6]), int(data[5]) + 10, int(data[6]) + 10), outline="red")
    imgDraw.rectangle((int(data[7]), int(data[8]), int(data[7]) + 10, int(data[8]) + 10), outline="red")
    imgDraw.rectangle((int(data[9]), int(data[10]), int(data[9]) + 10, int(data[10]) + 10), outline="red")
    img.show()

# for i, strs in enumerate(labdata):
#     if i < 2:
#         continue
#     print('strs',strs)
#     data = strs.strip().split()
#     print("data:",data)
#     img = Image.open(os.path.join(r"F:\anno","{}".format(data[0])))
#     imgDraw = ImageDraw.Draw(img)
#     imgDraw.rectangle((int(data[1]), int(data[2]), int(data[1])+int(data[3]), int(data[2])+int(data[4])),outline="red")
#     img.show()