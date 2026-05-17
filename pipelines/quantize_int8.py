import argparse, torch
import torch.ao.quantization as tq
from utils.data import get_loaders
from utils.metrics import accuracy
from models.resnet18 import ResNet18
from models.mobilenetv2 import MobileNetV2
from models.tinycnn import TinyCNN
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
    ap.add_argument('--ckpt', default='artifacts/resnet18_best.pt')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = 'cpu'  # quantized models run on CPU

    _, test_loader, classes = get_loaders(args.dataset, batch_size=128)
    net = get_model(args.model, classes)
    ckpt = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(ckpt['state_dict'])

    # dynamic quantization on Linear layers (portable baseline)
    net_q = tq.quantize_dynamic(net, {torch.nn.Linear}, dtype=torch.qint8)

    os.makedirs('artifacts', exist_ok=True)
    out_path = f'artifacts/quantized_{args.model}.pt'
    torch.save({'model': f'{args.model}_int8_dynamic', 'state_dict': net_q.state_dict()}, out_path)

    acc = accuracy(net_q, test_loader, device)
    print(f'INT8 dynamic accuracy: {acc:.2f}%')
    print('Saved', out_path)

if __name__ == '__main__':
    main()
