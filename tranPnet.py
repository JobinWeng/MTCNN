from train import trainer
from net import pNet

if __name__ == '__main__':
    net = pNet()
    train = trainer(net,'./model/Pnet.pt',[r"F:\jobin\MTCC数据集\SampleData05\12"])
    train.train()