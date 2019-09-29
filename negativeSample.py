import urllib.request
import urllib.parse
import re
import os
from PIL import  Image
import random

def GePictur():
    # 添加header，其中Referer是必须的,否则会返回403错误，User-Agent是必须的，这样才可以伪装成浏览器进行访问
    header = \
        {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
            "referer": "https://image.baidu.com"
        }
    url = "https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord={word}&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=0&word={word}&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&cg=girl&pn={pageNum}&rn=30&gsm=1e00000000001e&1490169411926="
    keyword = input("请输入搜索关键字：")
    # 转码
    keyword = urllib.parse.quote(keyword, 'utf-8')

    n = 0
    j = 0

    while (n < 10000):
        error = 0
        n += 30
        # url
        url1 = url.format(word=keyword, pageNum=str(n))
        # 获取请求
        rep = urllib.request.Request(url1, headers=header)
        # 打开网页
        rep = urllib.request.urlopen(rep)
        # 获取网页内容
        try:
            html = rep.read().decode('utf-8')
            # print(html)
        except:
            print("出错了！")
            error = 1
            print("出错页数：" + str(n))
        if error == 1:
            continue
        # 正则匹配
        p = re.compile("thumbURL.*?\.jpg")
        # 获取正则匹配到的结果，返回list
        s = p.findall(html)
        if os.path.isdir("F://pic") != True:
            os.makedirs("F://pic")
        with open("testpic.txt", "a") as f:
            # 获取图片
            for i in s:
                print(i)
                i = i.replace('thumbURL":"', '')
                print(i)
                f.write(i)
                f.write("\n")
                # 保存图片
                urllib.request.urlretrieve(i, "f://pic/pic{num}.jpg".format(num=j))
                j += 1
            f.close()
    print("总共爬取图片数为：" + str(j))

orignPath = r"F:\mtccLabel_negative\orign"
labelFilePath = r"F:\mtccLabel_negative"

def makeNegativeLabel(cropSize = 48, startIndex=0):
    fileList = os.listdir(orignPath)

    labelFile = open(os.path.join(labelFilePath,"{}.txt".format(cropSize)),'w')

    for index in fileList:
        img = Image.open(os.path.join(orignPath,index))

        if img.getbands() == ('R', 'G', 'B'):
            for _ in range(250):
                w,h =img.size
                x1 = random.randint(0, w-cropSize)
                y1 = random.randint(0, h-cropSize)
                x2 = x1 + cropSize
                y2 = y1 + cropSize

                cropImg = img.crop((x1,y1,x2,y2))
                cropImg.save(os.path.join(labelFilePath,str(cropSize),"{}.jpg".format(startIndex)))
                labelFile.write("{0}.jpg 0 0 0 0 0\n".format(startIndex))
                startIndex += 1


if __name__ == '__main__':
    makeNegativeLabel(12,0)