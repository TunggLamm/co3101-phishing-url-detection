# CO3101 - Đề tài C1: Phát hiện URL lừa đảo (Phishing URL Detection Audit)

- **Tên đề tài:** Phát hiện URL lừa đảo: bao nhiêu phần của con số 99 phần trăm là do cách dựng dữ liệu (*Phishing URL detection: how much of the reported 99 percent comes from dataset construction*).
- **Giảng viên hướng dẫn:** TS. Nguyễn An Khương
- **Nhóm 02:**
  - Khiếu Tùng Lâm (MSSV: 2452657) — Phụ trách miền bài toán, chọn đặc trưng, phân tích lỗi.
  - Lê Mai Minh Hiếu (MSSV: 2452331) — Phụ trách thiết kế thí nghiệm, hai sơ đồ chia dữ liệu và thống kê.

---

## 1. Chuẩn bị môi trường & Dữ liệu

1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
2. Tải bộ dữ liệu **PhiUSIIL Phishing URL Dataset** từ UCI và đặt vào thư mục `data/raw/` dưới tên `PhiUSIIL_Phishing_URL_Dataset.csv`:
   - Link tải: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
   - Đường dẫn: `data/raw/PhiUSIIL_Phishing_URL_Dataset.csv` (`235,795` dòng, `56` cột; quy ước chuẩn hóa nhãn trong dự án: `1 = Phishing`, `0 = Legitimate`).

---

## 2. Cấu trúc Kho mã

```text
├── data/raw/PhiUSIIL_Phishing_URL_Dataset.csv   # Bộ dữ liệu gốc từ UCI (bị chặn trên git)
├── src/
│   ├── data_split.py        # Chia ngẫu nhiên (random_split) & chia tách miền (etld1_split)
│   ├── features.py          # Kiểm toán 56 cột & trích xuất 10 đặc trưng từ vựng cốt lõi
│   ├── utils.py             # Tính PR-AUC, hạ tỉ lệ dương (adjust_positive_rate) & CI Student-t
│   ├── evaluation.py        # Khung đánh giá đơn/đa hạt giống (Mean ± 95% CI)
│   ├── run_splits.py        # Kiểm chứng và thống kê hai sơ đồ chia dữ liệu
│   ├── one_line_rule.py     # Baseline 1: Luật một dòng (tiền tố https://www.)
│   ├── train_lr.py          # Baseline 2: Logistic Regression trên 10 đặc trưng từ vựng
│   ├── train_gb.py          # Baseline 3: Gradient Boosting trên 10 đặc trưng từ vựng
│   ├── run_ablation.py      # Cắt giảm đặc trưng (50 → 46 → 18 → 10 cột) & kiểm toán rò rỉ đơn cột
│   └── test/
│       ├── test_required.py     # Tổng hợp 3 kiểm thử bắt buộc (Tiêu chí A3)
│       ├── test_data_split.py   # Unit tests cho extract_etld1 & chia dữ liệu không trùng miền
│       ├── test_utils.py        # Unit tests cho PR-AUC, hạ tỉ lệ dương & khoảng tin cậy 95%
│       └── test_evaluation.py   # Unit tests cho đánh giá đơn/đa hạt giống & ablation
└── requirements.txt
```

---

## 3. Chạy các Thí nghiệm & Baselines

Tại thư mục gốc của dự án, chạy các lệnh sau:

### 3.1. Kiểm tra phân chia dữ liệu & Thống kê 10 đặc trưng từ vựng
```bash
python src/run_splits.py
python src/features.py
```

### 3.2. Chạy 3 Baselines bắt buộc (`K = 5` hạt giống, `Mean ± 95% CI`)
Mỗi kịch bản tự động đánh giá đối chứng giữa **Random Split** và **eTLD+1 Split** qua 5 hạt giống `(42, 43, 44, 45, 46)`, kèm bảng đánh giá ở các mức tỉ lệ dương thực tế (`42.81%`, `10%`, `5%`, `1%`):

```bash
# Baseline 1: Luật một dòng (https://www.)
python src/one_line_rule.py

# Baseline 2: Logistic Regression (10 đặc trưng từ vựng)
python src/train_lr.py

# Baseline 3: Gradient Boosting (10 đặc trưng từ vựng)
python src/train_gb.py
```

### 3.3. Kiểm toán cắt giảm đặc trưng (Feature Ablation & Single-Feature Leakage Audit)
Đo độ tụt hiệu năng qua 4 cấp độ đặc trưng (`50 → 46 → 18 → 10` cột) và kiểm toán riêng từng cột rò rỉ (`IsHTTPS`, `URLSimilarityIndex`, `TLDLegitimateProb`, `URLCharProb`) trên phép chia `eTLD+1`:

```bash
python src/run_ablation.py
```

---

## 4. Chạy Kiểm thử Tự động (Unit Tests)

```bash
# Chạy toàn bộ bộ kiểm thử (21 unit tests)
pytest -v

# Chạy riêng 3 kiểm thử bắt buộc theo Mục 6 Phiếu đề tài (Tiêu chí A3)
pytest src/test/test_required.py -v
```
