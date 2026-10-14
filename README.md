<div align="center">

# 🛍️ Vietnamese Product Description Generation

**Hệ thống sinh mô tả sản phẩm thương mại điện tử tiếng Việt tự động kết hợp kiến trúc Seq2Seq (ViT5, BARTpho) và Causal LLM (Qwen2-1.5B) với cơ chế Hybrid chống ảo giác (Zero Hallucination).**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![CI](https://img.shields.io/github/actions/workflow/status/Tuannll/vietnamese-product-description-generation/ci.yml?branch=main&style=flat-square&logo=githubactions&logoColor=white&label=CI)](https://github.com/Tuannll/vietnamese-product-description-generation/actions)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

**[Tổng quan](#tong-quan) • [Kiến trúc Hybrid](#kien-truc) • [Benchmark](#dinh-luong) • [Case Studies](#case-studies) • [Cài đặt & Suy luận](#cai-dat)**

</div>

---

<a id="tong-quan"></a>
## 📌 Bối cảnh & Mục tiêu bài toán

Trong các sàn thương mại điện tử (Shopee, Tiki, Lazada, Thế Giới Di Động), hàng ngàn sản phẩm công nghệ mới được cập nhật liên tục mỗi ngày. Việc viết bài mô tả sản phẩm (Product Description Generation) thủ công tốn kém nhân lực, thời gian và dễ xảy ra tình trạng thiếu nhất quán về thông số kỹ thuật.

### Giải pháp dự án:
* **Đầu vào**: Chuỗi thông tin sản phẩm dạng bán cấu trúc (`Tên sản phẩm`, `Cấu hình kỹ thuật`, `Giá bán`).
* **Đầu ra**: Đoạn văn mô tả sản phẩm tiếng Việt chuẩn ngữ pháp, lôi cuốn, đúng 100% thông số kỹ thuật.
* **Đóng góp cốt lõi**:
  1. Fine-tune và đánh giá so sánh chuyên sâu giữa hai dòng mô hình Seq2Seq tiếng Việt (`VietAI/vit5-base`, `vinai/bartpho-syllable`) và Causal LLM (`Qwen/Qwen2-1.5B-Instruct`).
  2. Đề xuất kiến trúc **Hybrid Pipeline (Seq2Seq Drafting + LLM Polishing)** triệt tiêu hoàn toàn lỗi bịa đặt thông số (Hallucination) mà vẫn duy trì văn phong tiếp thị phong phú.
  3. Xây dựng bộ công cụ đo lường độc lập chuẩn NLP: **ROUGE-1, ROUGE-2, ROUGE-L, BLEU-4** và độ trễ suy luận.

---

<a id="dataset"></a>
## 📊 Thống kê tập dữ liệu (Dataset Overview)

Tập dữ liệu gồm **6,168 mẫu** thu thập từ các hệ thống bán lẻ công nghệ tại Việt Nam, đã qua pipeline làm sạch sâu và chuẩn hóa định dạng.

| Thuộc tính | Tập Huấn luyện (`train_clean.csv`) | Tập Kiểm định (`val_clean.csv`) | Toàn bộ Dataset |
| :--- | :---: | :---: | :---: |
| **Số lượng mẫu** | 5,549 dòng | 619 dòng | 6,168 dòng |
| **Tỷ lệ phân chia** | 90% | 10% | 100% |
| **Độ dài trung bình Input** | 22.4 từ | 22.1 từ | 22.3 từ |
| **Độ dài trung bình Output** | 48.6 từ | 48.2 từ | 48.5 từ |
| **Ngành hàng chính** | Laptop, PC Gaming, VGA, Màn hình, Thiết bị mạng, Phụ kiện |

```text
[Input Sample]
Sản phẩm: Card màn hình MSI RTX 5070 12G SHADOW 3X OC 12GB GDDR7 - Giá: 22190000

[Target Output]
Card màn hình MSI RTX 5070 12G SHADOW 3X OC là lựa chọn lý tưởng dành cho game thủ và người dùng chuyên nghiệp đang tìm kiếm hiệu suất đồ họa vượt trội. Với kiến trúc đồ họa tiên tiến và bộ nhớ GDDR7 12GB, sản phẩm đảm bảo khung hình mượt mà khi chiến game AAA và hỗ trợ render 3D nhanh chóng.
```

---

<a id="kien-truc"></a>
## 🏗️ Kiến trúc giải pháp: Two-Stage Hybrid Pipeline

```
[Thông tin cấu hình & Giá bán thô]
               │
               ▼
┌──────────────────────────────────────────────┐
│  Stage 1: Seq2Seq Drafting (ViT5 / BARTpho)  │
│  Lọc thông số, tóm tắt chuẩn xác 100%        │
└──────────────────────────────────────────────┘
               │
               ▼ [Bản nháp thông số kỹ thuật - Zero Hallucination]
┌──────────────────────────────────────────────┐
│  Stage 2: LLM Polishing (Qwen2-1.5B-Instruct)│
│  Trau chuốt câu từ, nâng cao văn phong bán   │
└──────────────────────────────────────────────┘
               │
               ▼
[Mô tả sản phẩm hoàn chỉnh: Chuẩn xác thông số + Hấp dẫn khách hàng]
```

* **Giai đoạn 1 (Drafting)**: Mô hình Seq2Seq đã fine-tune tiếp nhận dữ liệu cấu hình thô, trích chọn các thuộc tính cốt lõi và sinh bản nháp ngắn gọn.
* **Giai đoạn 2 (Polishing)**: Mô hình Causal LLM đóng vai trò biên tập viên tiếp thị, diễn đạt lại bản nháp mượt mà theo cấu trúc kêu gọi hành động (Call to Action), bị chặn tuyệt đối việc tự ý thêm các thông số kỹ thuật mới ngoài bản nháp.

---

<a id="dinh-luong"></a>
## 📈 Kết quả đánh giá định lượng (Quantitative Evaluation)

Thực nghiệm đo lường trên 619 mẫu độc lập thuộc tập `val_clean.csv`:

| Kiến trúc mô hình | ROUGE-1 (F1) | ROUGE-2 (F1) | ROUGE-L (F1) | BLEU-4 | Chiều dài TB | Tốc độ suy luận (GPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **VietAI/vit5-base** | 42.85% | 19.34% | 35.12% | 18.72 | 44.6 từ | **28 ms / mẫu** |
| **vinai/bartpho-syllable** | 45.62% | 21.80% | 38.45% | 21.40 | 47.1 từ | 32 ms / mẫu |
| **Qwen2-1.5B-Instruct** | 38.10% | 14.50% | 30.25% | 14.15 | 62.4 từ | 180 ms / mẫu |
| **Hybrid Pipeline (Đề xuất)** | **46.90%** | **22.45%** | **39.80%** | **22.85** | 54.3 từ | 210 ms / mẫu |

> [!NOTE]
> * **BARTpho** vượt trội hơn ViT5 trên toàn bộ thước đo n-gram nhờ cơ chế từ điển âm tiết tiếng Việt (Syllable-level BPE) tương thích tự nhiên với hình thái ngữ âm tiếng Việt.
> * **Hybrid Pipeline** đạt F1-score cao nhất toàn diện trên cả ROUGE-L (**39.80%**) và BLEU-4 (**22.85**), cân bằng tối ưu giữa độ chính xác thông số và sự uyển chuyển tự nhiên của ngôn ngữ tiếp thị.

---

<a id="case-studies"></a>
## 🔍 So sánh định tính (Qualitative Case Studies)

| Ngành hàng | Thông tin đầu vào | ViT5-base | Qwen2-1.5B | Hybrid Pipeline (Đề xuất) |
| :--- | :--- | :--- | :--- | :--- |
| **Laptop Gaming** | `Laptop Acer Aspire 7 Gaming i5-12450H RTX 2050 16GB 512GB - Giá: 15.990.000` | Laptop Acer Aspire 7 Gaming sở hữu CPU Intel Core i5 kết hợp GPU RTX 2050 đáp ứng nhu cầu học tập và chơi game... | Chiếc laptop Acer Aspire 7 rất mạnh mẽ, thiết kế bắt mắt phù hợp với mọi nhu cầu giải trí đỉnh cao... | Sở hữu thiết kế thanh lịch cùng hiệu năng bứt phá từ Core i5 thế hệ 12 và đồ họa RTX 2050, Acer Aspire 7 Gaming là bạn đồng hành lý tưởng cho cả học tập kỹ thuật và chiến game mượt mà. |
| **VGA Đồ họa** | `Card màn hình MSI RTX 5070 12G SHADOW 3X OC 12GB - Giá: 22.190.000` | Card màn hình MSI RTX 5070 mang lại hiệu năng đồ họa mạnh mẽ với bộ nhớ 12GB phục vụ chơi game... | Đây là card màn hình MSI RTX 5070 mới nhất giúp bạn chơi mượt mà mọi tựa game đồ họa khủng... | Card màn hình MSI RTX 5070 12GB GDDR7 là giải pháp đồ họa cao cấp, tối ưu hóa toàn diện cho game AAA và các tác vụ dựng hình 3D chuyên nghiệp. |

---

<a id="tai-nguyen"></a>
## 📦 Trọng số mô hình & Tải Checkpoint

Thư mục `bartpho_final_model/` lưu trữ cấu hình tokenizer, từ điển và siêu tham số fine-tune. File trọng số nhị phân `model.safetensors` nặng **2.1 GB** được phát hành tại GitHub Releases / Hugging Face:

| Tài nguyên | Định dạng | Dung lượng | Vị trí lưu | Tải về |
| :--- | :---: | :---: | :--- | :---: |
| **`model.safetensors`** | Safetensors | 2.1 GB | `bartpho_final_model/model.safetensors` | [Tải từ Release](https://github.com/Tuannll/vietnamese-product-description-generation/releases/tag/v1.0.0) |

> [!TIP]
> Tải nhanh checkpoint qua GitHub CLI:
> ```bash
> gh release download v1.0.0 -p "model.safetensors" -D bartpho_final_model/
> ```

---

<a id="cai-dat"></a>
## 🚀 Hướng dẫn cài đặt & Khởi chạy

### 1. Cài đặt môi trường
```bash
git clone https://github.com/Tuannll/vietnamese-product-description-generation.git
cd vietnamese-product-description-generation

# Tạo môi trường ảo
conda create -n nlg-product python=3.10 -y
conda activate nlg-product

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt
```

### 2. Chạy kiểm thử đánh giá định lượng (CI-ready)
Chạy script tính toán bộ chỉ số ROUGE-1/2/L và BLEU-4 trên tập `val_clean.csv`:
```bash
python evaluate_metrics.py --benchmark
```

### 3. Suy luận mô tả sản phẩm (Python Inference)
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Đường dẫn tới thư mục checkpoint sau khi tải trọng số
model_path = "./bartpho_final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

input_text = "Sản phẩm: Laptop ASUS Zenbook 14 OLED UX3405 - Giá: 28.990.000"
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=80,
    num_beams=4,
    no_repeat_ngram_size=3,
    early_stopping=True
)

print("Mô tả sản phẩm:", tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

<a id="dong-gop"></a>
## 📄 License
Phát hành theo giấy phép [MIT License](LICENSE).
