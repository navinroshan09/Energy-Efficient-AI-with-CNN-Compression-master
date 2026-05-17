import torch
import time

@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total

@torch.no_grad()
def latency(model, sample, repeat=1000, device="cpu"):
    model.eval()
    sample = sample.to(device)
    # warmup
    for _ in range(10):
        _ = model(sample)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        _ = model(sample)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.time() - t0
    ms = (dt / repeat) * 1000.0
    return ms, 1000.0 / ms  # latency ms, throughput img/s
