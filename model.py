import torch
from torch import nn

n_in = 28*28
#n_in = 123
n_mid = 512
n_out = 10
#n_out = 1

class MFNN1(nn.Module):
    def __init__(self):
        super(MFNN1, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu = nn.Sequential(
            nn.Linear(n_in, n_mid),
            nn.ReLU(),
            nn.Linear(n_mid, n_mid),
            nn.ReLU(),
            nn.Linear(n_mid, n_out),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu(x)
        return logits/n_mid



class MFNN2(nn.Module):
    def __init__(self):
        super(MFNN2, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu = nn.Sequential(
            nn.Linear(n_in, n_mid),
            nn.ReLU(),
            nn.Linear(n_mid, n_mid),
            nn.ReLU(),
            nn.Linear(n_mid, n_mid),
            nn.ReLU(),
            nn.Linear(n_mid, n_mid),
            nn.ReLU(),
            nn.Linear(n_mid, n_out),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu(x)
        return logits/n_mid

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu = nn.Sequential(
            nn.Linear(n_in, n_mid),
            nn.ReLU(),
            nn.Linear(n_mid, n_mid),
            nn.ReLU(),
            nn.Linear(n_mid, n_out)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu(x)
        return logits

class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.linear_relu = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1568, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 10),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.linear_relu(x)
        
        return x