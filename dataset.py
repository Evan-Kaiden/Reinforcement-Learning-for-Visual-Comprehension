from torchvision import datasets as dsets, transforms
from torch.utils.data import DataLoader, Subset
import torch

train_data = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Note: batch_size=1 is fine for episode-based training, but can be increased
trainloader = DataLoader(train_data, batch_size=1, shuffle=True)
testloader = DataLoader(test_data, batch_size=1, shuffle=False)