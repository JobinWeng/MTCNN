from train import trainer
from net import oNet

if __name__ == '__main__':
    net = oNet()
    train = trainer(net,'./model/Onet.pt',[r"F:\jobin\MTCC数据集\SampleData05\48"])
    train.train()