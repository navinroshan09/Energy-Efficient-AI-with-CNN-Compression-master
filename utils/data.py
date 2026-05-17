import torch
from torchvision import datasets, transforms

def get_loaders(name="cifar10", batch_size=128, num_workers=2):
    name = name.lower()
    if name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        train_t = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train = datasets.CIFAR10(root="./data", train=True, download=False, transform=train_t)
        test  = datasets.CIFAR10(root="./data", train=False, download=False, transform=test_t)
        classes = 10
    elif name == "mnist":
        train_t = transforms.Compose([transforms.ToTensor()])
        test_t  = transforms.Compose([transforms.ToTensor()])
        train = datasets.MNIST(root="./data", train=True, download=False, transform=train_t)
        test  = datasets.MNIST(root="./data", train=False, download=False, transform=test_t)
        classes = 10
    else:
        raise ValueError(f"Unknown dataset: {name}")
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, classes