
# ⚡ Energy-Efficient AI with CNN Compression

This project explores **energy-efficient neural network compression techniques** for deep learning models, with a focus on **ResNet-18** trained on **CIFAR-10**.  
We implement **Pruning, Quantization, and Knowledge Distillation**, benchmark trade-offs (accuracy vs size vs latency vs throughput), and provide a **Streamlit UI** for interactive exploration.  

---

## 📌 Features
- ✅ **Baseline Training**: Train ResNet-18 on CIFAR-10.  
- ✅ **Structured Pruning**: Remove redundant channels for efficiency.  
- ✅ **Quantization**: Convert FP32 → INT8 for faster inference.  
- ✅ **Knowledge Distillation**: Transfer knowledge from ResNet-18 (teacher) → MobileNetV2 (student).  
- ✅ **Benchmarking**: Compare accuracy, latency, size, and throughput.  
- ✅ **Interactive UI**: Upload/test images, visualize predictions, and compare models.  

---

## 📂 Project Structure
```
Energy-Efficient-AI-with-CNN-Compression/
│── demo/                 # Streamlit UI
│   └── app.py
│── pipelines/            # Training & compression pipelines
│   ├── train_baseline.py
│   ├── prune_structured.py
│   ├── quantize_dynamic.py
│   ├── distill.py
│   └── benchmark.py
│── models/               # CNN model definitions
│   ├── resnet18.py
│   └── mobilenetv2.py
│── utils/                # Helpers (data loaders, training loop, etc.)
│── artifacts/            # Trained & compressed models (.pt files)
│── reports/              # Benchmark results (CSV, charts)
│── prepare_data.py       # Download CIFAR-10 dataset
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation

### Step 1: Clone Repo
```bash
git clone https://github.com/iashokk/Energy-Efficient-AI-with-CNN-Compression.git
cd Energy-Efficient-AI-with-CNN-Compression
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv
# Activate it
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Linux/Mac
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download CIFAR-10 Dataset
```bash
python prepare_data.py
```

---

## 🚀 Usage

### 1. Train Baseline ResNet-18
```bash
python -m pipelines.train_baseline --dataset cifar10 --model resnet18 --epochs 25 --bs 128 --lr 0.001
```

### 2. Apply Structured Pruning
```bash
python -m pipelines.prune_structured --dataset cifar10 --model resnet18 --sparsity 0.5 --finetune-epochs 8 --ckpt artifacts/resnet18_best.pt
```

### 3. Apply Dynamic Quantization
```bash
python -m pipelines.quantize_int8 --ckpt artifacts/resnet18_best.pt
```

### 4. Knowledge Distillation (ResNet18 → MobileNetV2)
```bash
python -m pipelines.distill_kd --teacher resnet18 --student mobilenetv2 --epochs 20
```

### 5. Benchmark All Models
```bash
python -m pipelines.benchmark --dataset cifar10 --device cpu --repeat 500     --models artifacts/resnet18_best.pt artifacts/pruned_resnet18.pt artifacts/quantized_resnet18.pt artifacts/kd_mobilenetv2.pt
```

### 6. Run Streamlit UI
```bash
streamlit run demo/app.py
```

---

