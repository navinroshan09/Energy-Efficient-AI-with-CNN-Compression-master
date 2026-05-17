# --- Import path fix so utils/models are found when running Streamlit directly ---
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from utils.data import get_loaders
from models.resnet18 import ResNet18
from models.mobilenetv2 import MobileNetV2
from models.tinycnn import TinyCNN

# ------------------ App Config ------------------
st.set_page_config(page_title="Energy-Efficient AI Demo", layout="wide")
st.title("Energy-Efficient AI Demo")
st.caption("Compare accuracy, size, speed of compressed models on CIFAR-10. Try images or upload your own.")

# CIFAR-10 labels
CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# ------------------ Helpers ------------------
@st.cache_resource(show_spinner=False)
def load_test_loader(dataset="cifar10", bs=128):
    _, test_loader, classes = get_loaders(dataset, batch_size=bs,num_workers=0)
    return test_loader, classes

def instantiate_by_name(name: str, num_classes: int):
    n = (name or "resnet18").lower()
    if "mobilenet" in n:
        return MobileNetV2(num_classes)
    if "tinycnn" in n:
        return TinyCNN(num_classes)
    return ResNet18(num_classes)

def is_quantized_state_dict(state_dict: dict) -> bool:
    # heuristic: dynamic/static quantized layers have packed params/zero_point/scale
    for k in state_dict.keys():
        if "packed_params" in k or "zero_point" in k or "scale" in k:
            return True
    return False

@st.cache_resource(show_spinner=False)
def load_checkpoint(path, num_classes):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model_name = ckpt.get("model", os.path.basename(path))
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        model_name = os.path.basename(path)
        state_dict = ckpt
    else:
        model_name = "resnet18"
        state_dict = ckpt

    # instantiate
    m = instantiate_by_name(model_name, num_classes)

    # if looks quantized, quantize dynamically for Linear (portable baseline)
    if ("int8" in model_name.lower()) or is_quantized_state_dict(state_dict):
        import torch.ao.quantization as tq
        m = tq.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)

    missing, unexpected = m.load_state_dict(state_dict, strict=False)
    return m, model_name, missing, unexpected

def tensor_to_pil(x_denorm):
    # x_denorm: (3, H, W), in [0,1]
    arr = (x_denorm.numpy().transpose(1,2,0) * 255.0).clip(0,255).astype(np.uint8)
    return Image.fromarray(arr)

def denormalize_cifar(img_t):
    mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(1,3,1,1)
    std  = torch.tensor((0.2470, 0.2435, 0.2616)).view(1,3,1,1)
    return (img_t * std + mean).squeeze(0)

def preprocess_upload(img: Image.Image):
    # Resize to 32x32, convert to tensor + CIFAR10 normalization
    img = img.convert("RGB").resize((32,32))
    x = torch.from_numpy(np.asarray(img)).float().permute(2,0,1) / 255.0
    mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1)
    std  = torch.tensor((0.2470, 0.2435, 0.2616)).view(3,1,1)
    x = (x - mean) / std
    return x.unsqueeze(0)  # (1,3,32,32)

@st.cache_data(show_spinner=False)
def load_metrics_csv():
    path = os.path.join("reports", "metrics.csv")
    if os.path.isfile(path):
        try:
            df = pd.read_csv(path)
            # normalize names for pretty display
            df["model_ckpt"] = df["model_ckpt"].astype(str)
            return df
        except Exception:
            return None
    return None

def quick_latency(model, sample, repeat=200, device="cpu"):
    model.eval()
    sample = sample.to(device)
    with torch.no_grad():
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
        ms = (time.time() - t0) * 1000.0 / repeat
    return ms, 1000.0 / ms

# ------------------ Sidebar: Model selection & options ------------------
with st.sidebar:
    st.subheader("Model & Options")

    # find checkpoints
    ckpt_dir = "artifacts"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not ckpts:
        st.warning("No checkpoints found in artifacts/. Train models first.")
    ckpt_name = st.selectbox("Choose checkpoint", options=ckpts if ckpts else ["(none)"])

    dataset = st.selectbox("Dataset", ["cifar10"], index=0)
    device = "cpu"  # UI app runs on CPU by default for portability

    show_probs = st.checkbox("Show top-5 probabilities", value=True)
    show_metrics_table = st.checkbox("Show metrics table (reports/metrics.csv)", value=True)
    run_live_latency = st.checkbox("Measure latency now (quick)", value=False,
                                   help="If ON, times the selected model on a single image (repeat 200).")

col_left, col_mid, col_right = st.columns([1.2, 1.2, 1])

# ------------------ Load data & model ------------------
test_loader, num_classes = load_test_loader(dataset)
if ckpts:
    model_path = os.path.join(ckpt_dir, ckpt_name)
    with st.spinner(f"Loading {ckpt_name} ..."):
        model, model_tag, missing, unexpected = load_checkpoint(model_path, num_classes)
else:
    model = None
    model_tag = "(none)"

# ------------------ Section: Pick/Test CIFAR image ------------------
with col_left:
    st.subheader("Try a test image")
    idx = st.slider("Test image index", 0, 9999, 0)
    ds = test_loader.dataset
    img, label = ds[idx]
    x = img.unsqueeze(0)                 # (1,3,32,32)
    y = torch.tensor([label])
    x_denorm = denormalize_cifar(x.clone())
    pil_img = tensor_to_pil(x_denorm)
    st.image(pil_img, caption=f"Ground truth: {CIFAR10_CLASSES[y.item()]}", use_column_width=True)

# ------------------ Inference on selected test image ------------------
with col_mid:
    st.subheader("Prediction")
    if model is None:
        st.info("Train or place checkpoints in artifacts/ to test.")
    else:
        model.eval()
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_label = CIFAR10_CLASSES[pred_idx]
            st.markdown(f"**Model:** `{ckpt_name}`  \n**Predicted:** `{pred_label}`  \n**Truth:** `{CIFAR10_CLASSES[y.item()]}`")

            if show_probs:
                # Top-5
                top5_idx = np.argsort(probs)[-5:][::-1]
                top5_labels = [CIFAR10_CLASSES[i] for i in top5_idx]
                top5_vals = [float(probs[i]) for i in top5_idx]
                prob_df = pd.DataFrame({"class": top5_labels, "probability": top5_vals})
                st.bar_chart(prob_df.set_index("class"))

        if run_live_latency:
            ms, ips = quick_latency(model, x, repeat=200, device=device)
            st.caption(f"Live latency (single-image avg over 200 runs): **{ms:.2f} ms**  |  throughput **{ips:.1f} img/s**")

# ------------------ Upload your own image ------------------
with col_right:
    st.subheader("Upload an image")
    up = st.file_uploader("PNG/JPG (will be resized to 32×32, CIFAR-10 style)", type=["png","jpg","jpeg"])
    if up is not None and model is not None:
        img = Image.open(io.BytesIO(up.read()))
        x_u = preprocess_upload(img)  # (1,3,32,32)
        model.eval()
        with torch.no_grad():
            logits_u = model(x_u)
            p_u = F.softmax(logits_u, dim=1).cpu().numpy()[0]
            pred_u = CIFAR10_CLASSES[int(np.argmax(p_u))]
        st.image(img, caption=f"Your image → Predicted: {pred_u}", use_column_width=True)

# ------------------ Metrics / Comparison ------------------
st.markdown("---")
st.subheader("Model metrics & comparison")

metrics_df = load_metrics_csv()
if metrics_df is None or metrics_df.empty:
    st.info("No metrics file found at `reports/metrics.csv`. Run the benchmark script to generate it.")
else:
    # pretty formatting
    df_show = metrics_df.copy()
    for col in ["accuracy","size_mb","latency_ms","throughput_img_s"]:
        if col in df_show.columns:
            if col == "accuracy":
                df_show[col] = df_show[col].map(lambda v: f"{v:.2f}")
            elif col in ("size_mb", "latency_ms"):
                df_show[col] = df_show[col].map(lambda v: f"{v:.2f}")
            else:
                df_show[col] = df_show[col].map(lambda v: f"{v:.1f}")
    st.dataframe(df_show)

    # Pareto-ish scatter: latency vs accuracy, bubble size by file size
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        a = metrics_df["accuracy"].values
        l = metrics_df["latency_ms"].values
        s = metrics_df["size_mb"].values
        ax.scatter(l, a, s=np.clip(s, 5, 60)*5)
        for i, row in metrics_df.iterrows():
            ax.annotate(row["model_ckpt"], (row["latency_ms"], row["accuracy"]), fontsize=8)
        ax.set_xlabel("Latency (ms, lower is better)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy vs Latency (bubble ~ size MB)")
        st.pyplot(fig, clear_figure=True)
        
        # --- New Section: Sustainable AI Analysis ---
        st.subheader("Sustainable AI Analysis")
        st.markdown("**Accuracy Comparison**")
        fig2 = plt.figure(figsize=(8, 4))
        ax2 = fig2.add_subplot(111)
        # Assuming we want a bar chart comparing models
        x_labels = metrics_df["model_ckpt"].tolist()
        y_vals = metrics_df["accuracy"].astype(float).tolist()
        ax2.bar(x_labels, y_vals, color="#0088cc")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Model Accuracy Comparison")
        plt.xticks(rotation=45, ha='right')
        fig2.tight_layout()
        st.pyplot(fig2, clear_figure=True)
        
        # Add Model Size Comparison
        st.markdown("**Model Size Comparison**")
        fig3 = plt.figure(figsize=(8, 4))
        ax3 = fig3.add_subplot(111)
        s_vals = metrics_df["size_mb"].astype(float).tolist()
        ax3.bar(x_labels, s_vals, color="#28a745")
        ax3.set_ylabel("Size (MB) - Lower is better")
        ax3.set_title("Model Size Comparison")
        plt.xticks(rotation=45, ha='right')
        fig3.tight_layout()
        st.pyplot(fig3, clear_figure=True)
        
        # Add Latency Comparison
        st.markdown("**Latency Comparison**")
        fig4 = plt.figure(figsize=(8, 4))
        ax4 = fig4.add_subplot(111)
        l_vals = metrics_df["latency_ms"].astype(float).tolist()
        ax4.bar(x_labels, l_vals, color="#ffc107")
        ax4.set_ylabel("Latency (ms) - Lower is better")
        ax4.set_title("Model Latency Comparison")
        plt.xticks(rotation=45, ha='right')
        fig4.tight_layout()
        st.pyplot(fig4, clear_figure=True)
        
        # Add Throughput Comparison
        st.markdown("**Throughput Comparison**")
        fig5 = plt.figure(figsize=(8, 4))
        ax5 = fig5.add_subplot(111)
        t_vals = metrics_df["throughput_img_s"].astype(float).tolist()
        ax5.bar(x_labels, t_vals, color="#dc3545")
        ax5.set_ylabel("Throughput (img/s) - Higher is better")
        ax5.set_title("Model Throughput Comparison")
        plt.xticks(rotation=45, ha='right')
        fig5.tight_layout()
        st.pyplot(fig5, clear_figure=True)
        
    except Exception as e:
        st.caption(f"(Plot unavailable: {e})")

# ------------------ Footer Notes ------------------
with st.expander("What am I looking at? (Quick guide)"):
    st.markdown("""
- **Choose checkpoint:** pick baseline, pruned, quantized, or distilled models saved in `artifacts/`.
- **Try a test image:** select any sample from CIFAR-10 test split and see predicted label.
- **Upload an image:** test your own photo (UI resizes to 32×32 and normalizes like CIFAR-10).
- **Metrics table:** shows accuracy, file size, latency, throughput from `reports/metrics.csv` (run the benchmark script to generate).
- **Bubble chart:** trade-off between **accuracy (y)** and **latency (x)**; bubble size ≈ model file size.
- Tip: Quantized models should be faster; pruned/distilled models should be smaller.
""")
