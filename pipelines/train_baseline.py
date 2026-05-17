import argparse, torch, torch.nn as nn, torch.optim as optim
from utils.data import get_loaders
from utils.metrics import accuracy
from models.resnet18 import ResNet18
from models.mobilenetv2 import MobileNetV2
from models.tinycnn import TinyCNN
from tqdm import tqdm
import os

def get_model(name, num_classes):
    name = name.lower()
    if name == 'resnet18':
        return ResNet18(num_classes)
    if name == 'mobilenetv2':
        return MobileNetV2(num_classes)
    if name == 'tinycnn':
        return TinyCNN(num_classes)
    raise ValueError(f'Unknown model {name}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='cifar10')
    ap.add_argument('--model', default='resnet18')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, test_loader, classes = get_loaders(args.dataset, batch_size=args.bs)
    net = get_model(args.model, classes).to(device)
    opt = optim.Adam(net.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best = 0.0
    os.makedirs('artifacts', exist_ok=True)
    for ep in range(1, args.epochs+1):
        net.train()
        pbar = tqdm(train_loader, desc=f'Epoch {ep}/{args.epochs}')
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(net(x), y)
            loss.backward()
            opt.step()
        acc = accuracy(net, test_loader, device)
        print(f'Val Acc: {acc:.2f}%')
        if acc > best:
            best = acc
            path = f'artifacts/{args.model}_best.pt'
            torch.save({'model': args.model, 'state_dict': net.state_dict(), 'acc': best}, path)
            print('Saved:', path)

if __name__ == '__main__':
    main()
