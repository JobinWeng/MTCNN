import torch
import torch.nn as nn

class pNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=10,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.PReLU(),
        )

        self.con = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

        self.offset = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1, stride=1)

        self.landmark = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=1, stride=1)

    def forward(self, x):
        preOut = self.pre(x)
        Con = self.con(preOut)
        offset = self.offset(preOut)
        landmark = self.landmark(preOut)

        return Con, offset, landmark

class rNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, stride=1,padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=28, out_channels=48, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=2, stride=1),
            nn.PReLU(),
        )

        self.liner_0 = nn.Sequential(
            nn.Linear(in_features=3 * 3 * 64, out_features=128),
            nn.PReLU()
        )
        self.con = nn.Sequential(
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )
        self.offset = nn.Linear(in_features=128, out_features=4)
        self.landMark = nn.Linear(in_features=128, out_features=10)


    def forward(self, x):
        preOut = self.pre(x)

        liner_0 = self.liner_0(preOut.view(preOut.size(0),-1))
        con = self.con(liner_0)
        offset = self.offset(liner_0)
        landmark = self.landMark(liner_0)
        return con,offset,landmark

class oNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1,padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1),
            nn.PReLU()
        )
        self.liner_0 = nn.Sequential(
            nn.Linear(in_features=128 * 3 * 3, out_features=256),
            nn.PReLU()
        )
        self.con = nn.Sequential(
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )
        self.offset = nn.Linear(in_features=256, out_features=4)
        self.landmark = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        preOut = self.pre(x)
        liner_0 = self.liner_0(preOut.view(preOut.size(0),-1))
        con = self.con(liner_0)
        offset = self.offset(liner_0)
        landmark = self.landmark(liner_0)
        return con,offset,landmark

if __name__ == '__main__':
    a = torch.ones((1,3,48,48))
    # o = oNet()
    # print(o(a))
