from train import trainer
from net import rNet

if __name__ == '__main__':
    net = rNet()
    train = trainer(net,'./model/Rnet.pt',[r"F:\jobin\MTCC数据集\SampleData05\24"])
    train.train()