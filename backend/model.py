import torch
import torch.nn as nn
torch.manual_seed(42)

class ANN(nn.Module):
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
            
        )

        # flatten layer to reshape into a 1D vector to pass into the network
        # two hidden layers to help with learning complex representations
        # use dropout layer to help with regularisation
        # # No softmax layer as pytorch cross-entropy automatically applies it.
    
    def forward(self, x):
        return self.network(x)

class CNN(nn.Module):
    def __init__(self, num_class):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_class),
            # No softmax layer as pytorch cross-entropy automatically applies it.
        )

    def forward(self,x):
        x = self.conv_layers(x)
        return self.fc_layers(x)