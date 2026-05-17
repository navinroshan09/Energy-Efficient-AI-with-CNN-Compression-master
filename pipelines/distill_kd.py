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

def kd_loss_fn(student_logits, teacher_logits, y, alpha=0.7, T=4.0):
    ce = nn.CrossEntropyLoss()(student_logits, y)
    kl = nn.KLDivLoss(reduction='batchmean')(
        nn.LogSoftmax(dim=1)(student_logits / T),
        nn.Softmax(dim=1)(teacher_logits / T)
    ) * (T*T)
    return alpha * kl + (1 - alpha) * ce

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='cifar10')
    ap.add_argument('--teacher', default='resnet18')
    ap.add_argument('--student', default='mobilenetv2')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--alpha', type=float, default=0.7)
    ap.add_argument('--temperature', type=float, default=4.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--teacher_ckpt', default='artifacts/resnet18_best.pt')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, test_loader, classes = get_loaders(args.dataset, batch_size=args.bs)
    # load teacher
    teacher = get_model(args.teacher, classes).to(device)
    ckpt = torch.load(args.teacher_ckpt, map_location=device)
    teacher.load_state_dict(ckpt['state_dict'])
    teacher.eval()

    # student
    student = get_model(args.student, classes).to(device)
    opt = optim.Adam(student.parameters(), lr=args.lr)

    best = 0.0
    os.makedirs('artifacts', exist_ok=True)
    for ep in range(1, args.epochs+1):
        student.train()
        pbar = tqdm(train_loader, desc=f'KD Epoch {ep}/{args.epochs}')
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                tlog = teacher(x)
            slog = student(x)
            loss = kd_loss_fn(slog, tlog, y, alpha=args.alpha, T=args.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()

        acc = accuracy(student, test_loader, device)
        print(f'Val Acc: {acc:.2f}%')
        if acc > best:
            best = acc
            out = f'artifacts/kd_{args.student}.pt'
            torch.save({'model': args.student, 'state_dict': student.state_dict(), 'acc': best}, out)
            print('Saved:', out)

if __name__ == '__main__':
    main()
