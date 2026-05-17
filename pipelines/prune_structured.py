import argparse, torch, torch.nn.utils.prune as prune, torch.nn as nn
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.resnet18 import ResNet18
from utils.data import get_loaders
from utils.metrics import accuracy
from tqdm import tqdm
import os

def structured_channel_prune(model, amount=0.5):
    # prune Conv2d output channels (structured) by L1 norm per layer
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels > 4:
            try:
                prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
                prune.remove(module, 'weight')
            except Exception as e:
                print('Skip prune on', name, e)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='cifar10')
    ap.add_argument('--model', default='resnet18')
    ap.add_argument('--sparsity', type=float, default=0.5)
    ap.add_argument('--finetune-epochs', type=int, default=8)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ckpt', default='artifacts/resnet18_best.pt')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, test_loader, classes = get_loaders(args.dataset, batch_size=args.bs)
    net = ResNet18(classes).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(ckpt['state_dict'])

    net = structured_channel_prune(net, amount=args.sparsity)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best = 0.0
    os.makedirs('artifacts', exist_ok=True)
    for ep in range(1, args.finetune_epochs+1):
        net.train()
        for x, y in tqdm(train_loader, desc=f'Finetune {ep}/{args.finetune_epochs}'):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(net(x), y)
            loss.backward()
            opt.step()
        acc = accuracy(net, test_loader, device)
        print('Val Acc:', acc)
        if acc > best:
            best = acc
            torch.save({'model': 'resnet18_pruned', 'state_dict': net.state_dict(), 'acc': best},
                       'artifacts/pruned_resnet18.pt')
            print('Saved artifacts/pruned_resnet18.pt')

if __name__ == '__main__':
    main()
