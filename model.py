import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

class ANN(nn.module):
    def __init__(self, input_size, num_class):
        super(ANN, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,num_class),
            nn.Softmax(dim=1) 
        )

        # flatten layer to reshape into a 1D vector to pass into the network
        # two hidden layers to help with learning complex representations
        # use dropout layer to help with regularisation
        # use softmax for multi-class classification (digits from 0-9)
    
    def forward(self, x):
        return self.network(x)