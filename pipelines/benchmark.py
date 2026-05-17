import argparse, os, torch
import torch.nn as nn
import torch.ao.quantization as tq
from utils.data import get_loaders
from utils.metrics import accuracy, latency
from models.resnet18 import ResNet18
from models.mobilenetv2 import MobileNetV2
from models.tinycnn import TinyCNN

def _instantiate_from_name(name: str, num_classes: int):
    n = (name or "resnet18").lower()
    if "mobilenet" in n:
        return MobileNetV2(num_classes)
    if "tinycnn" in n:
        return TinyCNN(num_classes)
    return ResNet18(num_classes)

def _is_quantized_state_dict(state_dict: dict) -> bool:
    # Heuristic: quantized dynamic linear layers have packed params/zero_point/scale keys
    for k in state_dict.keys():
        if "packed_params" in k or "zero_point" in k or "scale" in k:
            return True
    return False

def load_model_from_ckpt(path, num_classes):
    ckpt = torch.load(path, map_location="cpu")

    # Extract (model_name, state_dict)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model_name = ckpt.get("model", "resnet18")
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        # raw state dict (already)
        model_name = os.path.basename(path)
        state_dict = ckpt
    else:
        # unknown format
        model_name = "resnet18"
        state_dict = ckpt

    m = _instantiate_from_name(model_name, num_classes)

    # If this looks like a quantized checkpoint, quantize model before loading
    needs_int8 = ("int8" in model_name.lower()) or _is_quantized_state_dict(state_dict)
    if needs_int8:
        m = tq.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)

    # Load with strict=False so minor head naming diffs don’t crash
    m.load_state_dict(state_dict, strict=False)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='cifar10')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--repeat', type=int, default=200)  # faster first run
    ap.add_argument('--models', nargs='+', required=True)
    args = ap.parse_args()

    dev = args.device
    _, test_loader, classes = get_loaders(args.dataset, batch_size=128)
    sample = next(iter(test_loader))[0][:1]  # one image

    rows = []
    for path in args.models:
        m = load_model_from_ckpt(path, classes).to(dev)
        acc = accuracy(m, test_loader, dev)
        ms, ips = latency(m, sample, repeat=args.repeat, device=dev)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        rows.append((os.path.basename(path), acc, size_mb, ms, ips))
        print(f'{path}: acc={acc:.2f} sizeMB={size_mb:.2f} lat_ms={ms:.3f} ips={ips:.1f}')

    # write CSV
    import csv
    os.makedirs('reports', exist_ok=True)
    with open('reports/metrics.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model_ckpt','accuracy','size_mb','latency_ms','throughput_img_s'])
        w.writerows(rows)
    print('Wrote reports/metrics.csv')

if __name__ == '__main__':
    main()
