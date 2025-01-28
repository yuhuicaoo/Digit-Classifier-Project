import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
torch.manual_seed(42)

def load_data(batch_size):
    train_dset = dsets.MNIST(
        root = "./data",
        train = True,
        transform = transforms.ToTensor(),
        download = True
    )

    test_dset = dsets.MNIST(
        root = "./data",
        train = False,
        transform = transforms.ToTensor()
    )

    # Prepare training , validation and testing data loader
    train_subset, val_subset = random_split(train_dset, [0.9, 0.1])

    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader