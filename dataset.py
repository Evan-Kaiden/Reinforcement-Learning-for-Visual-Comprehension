from torch.utils.data import DataLoader
from torchvision import datasets as dsets, transforms

train_tf = transforms.Compose([
    # transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.7, 1.3)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
])

test_tf = transforms.Compose([
    # transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.7, 1.3)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
])

train_data = dsets.CIFAR10(root='./data', train=True, download=True, transform=train_tf)
test_data = dsets.CIFAR10(root='./data', train=False, download=True, transform=test_tf)

trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
testloader = DataLoader(test_data, batch_size=32, shuffle=False)