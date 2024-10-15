# import
from torch import nn

''' model '''
# network
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.fc1 = nn.Linear(in_features=2048, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=10)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.droput = nn.Dropout(p=0.3)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.droput(self.pool(self.relu(self.conv1(x))))
        x = self.droput(self.pool(self.relu(self.conv2(x))))
        x = self.droput(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.droput(self.relu(self.fc1(x)))
        x = self.droput(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
'''
    입력    (1, 3, 32, 32)          = (batch, channels, H, W)

    합성곱1 -> (1, 32, 30, 30)      필터 (32, 3, 3, 3) = (out_ch, in_ch, H, W)
    풀링    -> (1, 32, 15, 15)

    합성곱2 -> (1, 64, 13, 13)      필터 (64, 32, 3, 3)
    풀링    -> (1, 64, 6, 6)

    합성곱3 -> (1, 128, 4, 4)        필터 (128, 64, 3, 3)

    평탄화  -> (1, 2048)

    밀집층1 -> (1, 512)
    밀집층2 -> (1, 256)
    밀집층3 -> (1, 10)
'''

# load
def load_model(device):
    model = CNN()
    model = model.to(device)

    return model

# test
# import torch
# m = CNN()
# x = torch.ones(1, 3, 32, 32)

# print(m.fc3(m.fc2(m.fc1(m.flatten(m.conv5(m.conv4(m.conv3(m.conv2(m.conv1(x))))))))).shape)