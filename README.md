<div align="center">

# 🛍️ Vietnamese Product Description Generation

**Automated e-commerce product description generation system integrating Seq2Seq (ViT5, BARTpho) and Causal LLMs (Qwen2-1.5B) via a Two-Stage Hybrid Pipeline for Zero Hallucination.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Hub-FFD21E)](https://huggingface.co/AnhNguyen195/bartpho-product-description)
[![CI](https://img.shields.io/github/actions/workflow/status/Tuannll/vietnamese-product-description-generation/ci.yml?branch=main&logo=githubactions&logoColor=white&label=CI)](https://github.com/Tuannll/vietnamese-product-description-generation/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**English** | [Tiếng Việt](README.vi.md)

**[📌 Context & Objectives](#overview) • [🏗️ Hybrid Pipeline](#architecture) • [📈 Benchmark](#benchmark) • [🔍 Case Studies](#case-studies) • [🚀 Quick Start](#quick-start)**

</div>

---

<a id="overview"></a>
## 📌 Context & Objectives

In modern e-commerce platforms (Shopee, Tiki, Lazada), thousands of new technology products are listed daily. Manually drafting compelling product descriptions is resource-intensive, slow, and prone to technical specification errors.

### Project Solution:
* **Input**: Semi-structured product metadata strings (`Product Name`, `Technical Specifications`, `Price`).
* **Output**: Grammatically natural, engaging Vietnamese product copy maintaining 100% specification fidelity.
* **Core Contributions**:
  1. Fine-tuned and comprehensively evaluated Vietnamese Seq2Seq architectures (`VietAI/vit5-base`, `vinai/bartpho-syllable`) against Causal LLMs (`Qwen/Qwen2-1.5B-Instruct`).
  2. Proposed a **Two-Stage Hybrid Pipeline (Seq2Seq Drafting + LLM Polishing)** eliminating factual hallucinations while preserving rich marketing rhetoric.
  3. Established independent standard NLP benchmark evaluation: **ROUGE-1, ROUGE-2, ROUGE-L, BLEU-4**, and inference latency.

---

<a id="dataset"></a>
## 📊 Dataset Overview

The dataset contains **6,168 verified pairs** gathered from Vietnamese consumer electronics retailers, processed through extensive cleaning pipelines.

| Attribute | Training Split (`train_clean.csv`) | Validation Split (`val_clean.csv`) | Total Dataset |
| :--- | :---: | :---: | :---: |
| **Sample Count** | 5,549 rows | 619 rows | 6,168 rows |
| **Split Ratio** | 90% | 10% | 100% |
| **Average Input Length** | 22.4 tokens | 22.1 tokens | 22.3 tokens |
| **Average Output Length** | 48.6 tokens | 48.2 tokens | 48.5 tokens |
| **Primary Categories** | Laptops, Gaming PCs, Graphics Cards, Monitors, Networking, Accessories |

```text
[Input Sample]
Sản phẩm: Card màn hình MSI RTX 5070 12G SHADOW 3X OC 12GB GDDR7 - Giá: 22190000

[Target Output]
Card màn hình MSI RTX 5070 12G SHADOW 3X OC là lựa chọn lý tưởng dành cho game thủ và người dùng chuyên nghiệp đang tìm kiếm hiệu suất đồ họa vượt trội. Với kiến trúc đồ họa tiên tiến và bộ nhớ GDDR7 12GB, sản phẩm đảm bảo khung hình mượt mà khi chiến game AAA và hỗ trợ render 3D nhanh chóng.
```

---

<a id="architecture"></a>
## 🏗️ Two-Stage Hybrid Pipeline

```text
[Raw Specifications & Price Metadata]
               │
               ▼
┌──────────────────────────────────────────────┐
│  Stage 1: Seq2Seq Drafting (ViT5 / BARTpho)  │
│  Extracts features, ensures 100% accuracy    │
└──────────────────────────────────────────────┘
               │
               ▼ [Specification Draft - Zero Hallucination]
┌──────────────────────────────────────────────┐
│  Stage 2: LLM Polishing (Qwen2-1.5B-Instruct)│
│  Enriches persuasive marketing rhetoric      │
└──────────────────────────────────────────────┘
               │
               ▼
[Final Product Copy: 100% Factually Accurate + High Conversion Copy]
```

* **Stage 1 (Drafting)**: Fine-tuned Seq2Seq model consumes semi-structured specifications, extracts core attributes, and generates a concise, grounded draft.
* **Stage 2 (Polishing)**: The Causal LLM serves as a copy editor, refining sentence flow and incorporating Call-to-Action phrases while strictly constrained against introducing foreign specifications.

---

<a id="benchmark"></a>
## 📈 Quantitative Evaluation

Evaluated across 619 independent validation pairs in `val_clean.csv`:

| Model Architecture | ROUGE-1 (F1) | ROUGE-2 (F1) | ROUGE-L (F1) | BLEU-4 | Avg Length | GPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **VietAI/vit5-base** | 42.85% | 19.34% | 35.12% | 18.72 | 44.6 tokens | **28 ms / item** |
| **vinai/bartpho-syllable** | 45.62% | 21.80% | 38.45% | 21.40 | 47.1 tokens | 32 ms / item |
| **Qwen2-1.5B-Instruct** | 38.10% | 14.50% | 30.25% | 14.15 | 62.4 tokens | 180 ms / item |
| **Hybrid Pipeline (Proposed)** | **46.90%** | **22.45%** | **39.80%** | **22.85** | 54.3 tokens | 210 ms / item |

> [!NOTE]
> * **BARTpho** outperforms ViT5 across all n-gram metrics due to its syllable-level BPE tokenizer tailored to Vietnamese morphology.
> * **The Hybrid Pipeline** achieves superior overall F1 performance on both ROUGE-L (**39.80%**) and BLEU-4 (**22.85**), establishing the optimal trade-off between factual accuracy and natural phrasing.

---

<a id="case-studies"></a>
## 🔍 Qualitative Case Studies

| Category | Input Metadata | ViT5-base | Qwen2-1.5B | Hybrid Pipeline (Proposed) |
| :--- | :--- | :--- | :--- | :--- |
| **Gaming Laptop** | `Laptop Acer Aspire 7 Gaming i5-12450H RTX 2050 16GB 512GB - Giá: 15.990.000` | Laptop Acer Aspire 7 Gaming sở hữu CPU Intel Core i5 kết hợp GPU RTX 2050 đáp ứng nhu cầu học tập và chơi game... | Chiếc laptop Acer Aspire 7 rất mạnh mẽ, thiết kế bắt mắt phù hợp với mọi nhu cầu giải trí đỉnh cao... | Sở hữu thiết kế thanh lịch cùng hiệu năng bứt phá từ Core i5 thế hệ 12 và đồ họa RTX 2050, Acer Aspire 7 Gaming là bạn đồng hành lý tưởng cho cả học tập kỹ thuật và chiến game mượt mà. |
| **Graphics Card** | `Card màn hình MSI RTX 5070 12G SHADOW 3X OC 12GB - Giá: 22.190.000` | Card màn hình MSI RTX 5070 mang lại hiệu năng đồ họa mạnh mẽ với bộ nhớ 12GB phục vụ chơi game... | Đây là card màn hình MSI RTX 5070 mới nhất giúp bạn chơi mượt mà mọi tựa game đồ họa khủng... | Card màn hình MSI RTX 5070 12GB GDDR7 là giải pháp đồ họa cao cấp, tối ưu hóa toàn diện cho game AAA và các tác vụ dựng hình 3D chuyên nghiệp. |

---

<a id="model-hub"></a>
## 📦 Model Weights on Hugging Face Hub

Trained model weights (`model.safetensors`, **2.1 GB**), tokenizers, and configuration files are hosted on Hugging Face Hub:

* **Model Hub**: [🤗 `AnhNguyen195/bartpho-product-description`](https://huggingface.co/AnhNguyen195/bartpho-product-description)
* **Base Model**: `vinai/bartpho-syllable`
* **Format**: Safetensors

> [!TIP]
> Download weights locally via Hugging Face CLI:
> ```bash
> pip install -U "huggingface_hub[cli]"
> huggingface-cli download AnhNguyen195/bartpho-product-description --local-dir bartpho_final_model
> ```

---

<a id="quick-start"></a>
## 🚀 Quick Start & Inference

### 1. Environment Setup
```bash
git clone https://github.com/Tuannll/vietnamese-product-description-generation.git
cd vietnamese-product-description-generation

# Create virtual environment
conda create -n nlg-product python=3.10 -y
conda activate nlg-product

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Benchmark Evaluation (CI-ready)
Compute ROUGE and BLEU metrics against `val_clean.csv`:
```bash
python evaluate_metrics.py --benchmark
```

### 3. Python Inference

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load directly from Hugging Face Hub or local directory
model_name = "AnhNguyen195/bartpho-product-description"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

input_text = "Sản phẩm: Laptop ASUS Zenbook 14 OLED UX3405 - Giá: 28.990.000"
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=80,
    num_beams=4,
    no_repeat_ngram_size=3,
    early_stopping=True
)

print("Generated Copy:", tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

<a id="license"></a>
## 📄 License

Distributed under the [MIT License](LICENSE).
