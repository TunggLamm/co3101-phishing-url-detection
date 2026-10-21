# CO3101 - Đề tài C1: Phát hiện URL lừa đảo (Phishing URL Detection Audit)


- **Tên đề tài:** Phát hiện URL lừa đảo: bao nhiêu phần của con số 99 phần trăm là do cách dựng dữ liệu (*Phishing URL detection: how much of the reported 99 percent comes from dataset construction*).
- **Giảng viên hướng dẫn:** TS. Nguyễn An Khương
- **Nhóm 02:**
  - Khiếu Tùng Lâm (MSSV: 2452657) — Phụ trách miền bài toán, chọn đặc trưng, phân tích lỗi.
  - Lê Mai Minh Hiếu (MSSV: 2452331) — Phụ trách thiết kế thí nghiệm, hai sơ đồ chia dữ liệu và thống kê.

## 1. Chuẩn bị môi trường & Dữ liệu
1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
2. Tải data set từ web chính chủ của UCI và đặt vào data/raw dưới tên "PhiUSIIL_Phishing_URL_Dataset.csv"
   Link download: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
