import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os

def prepare_cifar10():
    root = "./data"
    if not os.path.isdir(root):
        os.makedirs(root, exist_ok=True)

    transform = transforms.ToTensor()

    print("📥 Downloading CIFAR-10 training set...")
    datasets.CIFAR10(root=root, train=True, download=True, transform=transform)

    print("📥 Downloading CIFAR-10 test set...")
    datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    print("✅ CIFAR-10 dataset is ready at", root)

if __name__ == "__main__":
    prepare_cifar10()
