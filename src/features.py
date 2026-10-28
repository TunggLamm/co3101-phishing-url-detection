from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"

#ĐỊNH NGHĨA CÁC NHÓM CỘT BỊ LOẠI
ID_AND_LABEL_COLS = ["FILENAME", "URL", "label"]

LEAKAGE_AND_RAW_COLS = [
    "Domain",              # dùng để chia eTLD+1
    "TLD",                 # chuỗi thô
    "IsHTTPS",             # nhiễu khuôn mẫu (100% mẫu hợp lệ có https://www)
    "URLSimilarityIndex",  # rò rỉ nhãn
    "TLDLegitimateProb",   # rò rỉ nhãn
    "URLCharProb",         # rò rỉ nhãn
]

REDUNDANT_LEXICAL_COLS = [
    "TLDLength",           # ít thông tin phân loại
    "NoOfObfuscatedChar",  # trùng lặp với HasObfuscation
    "ObfuscationRatio",    # trùng lặp với HasObfuscation
    "NoOfLettersInURL",    # đa cộng tuyến với URLLength & LetterRatioInURL
    "NoOfDegitsInURL",     # đa cộng tuyến với URLLength & LetterRatioInURL
    "CharContinuationRate",# phụ thuộc cấu trúc chuỗi con
    "NoOfAmpersandInURL",  # đa cộng tuyến với NoOfEqualsInURL
    "NoOfOtherSpecialCharsInURL", # đã được đại diện bởi SpacialCharRatioInURL
]

# DANH SÁCH 10 ĐẶC TRƯNG TỪ VỰNG ĐƯỢC CHỌN
SELECTED_LEXICAL_FEATURES = [
    "URLLength",           # Tổng độ dài chuỗi URL
    "DomainLength",        # Độ dài phần tên miền
    "IsDomainIP",          # Tên miền có phải địa chỉ IP trần (0/1)
    "NoOfSubDomain",       # Số lượng subdomain
    "HasObfuscation",      # Có mã hóa/làm rối ký tự trên URL (0/1)
    "LetterRatioInURL",    # Tỉ lệ chữ cái trên tổng độ dài URL
    "DegitRatioInURL",     # Tỉ lệ chữ số trên tổng độ dài URL
    "NoOfEqualsInURL",     # Số lượng dấu =
    "NoOfQMarkInURL",       # Số lượng dấu ?
    "SpacialCharRatioInURL", # Tỉ lệ ký tự đặc biệt trên tổng độ dài URL
]

# GIẢI THÍCH Ý NGHĨA 10 ĐẶC TRƯNG
FEATURE_DESCRIPTIONS = {
    "URLLength": "Tổng độ dài chuỗi URL",
    "DomainLength": "Độ dài phần tên miền",
    "IsDomainIP": "Tên miền là địa chỉ IP (0/1)",
    "NoOfSubDomain": "Số lượng tên miền phụ (subdomain)",
    "HasObfuscation": "Có làm rối/mã hóa ký tự (0/1)",
    "LetterRatioInURL": "Tỉ lệ chữ cái trên URL",
    "DegitRatioInURL": "Tỉ lệ chữ số trên URL",
    "NoOfEqualsInURL": "Số lượng dấu bằng '='",
    "NoOfQMarkInURL": "Số lượng dấu chấm hỏi '?'",
    "SpacialCharRatioInURL": "Tỉ lệ ký tự đặc biệt trên URL",
}

def select_lexical_features(df: pd.DataFrame):
    """
    Hàm dùng chung cho các module huấn luyện (Logistic Regression / Gradient Boosting):
    - Đầu vào: DataFrame gốc độc từ CSV PhiUSIIL
    - Đầu ra:
        + X (pd.DataFrame): Ma trận chỉ gồm 10 cột đặc trưng từ vựng cốt lõi
        + y (pd.Series): Nhãn đã chuẩn hóa (1 = Phishing, 0 = Legitimate)
    """

    # kiểm tra xem có thiếu cột nào trong 10 cột bắt buộc không
    missing_cols = set(SELECTED_LEXICAL_FEATURES) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Thiếu các cột đặc trưng bắt buộc: {missing_cols}")

    if "label" not in df.columns:
        raise ValueError("Không tìm thấy cột 'label' trong DataFrame.")

    # trích xuất đúng 10 cột đặc trưng
    X = df[SELECTED_LEXICAL_FEATURES].copy()

    # đảm bảo không có giá trị trống
    if X.isna().any().any():
        raise ValueError("Phát hiện giá trị NaN trong 10 cột đặc trưng từ vựng")

    # chuẩn hóa nhãn
    y = (df["label"] == 0).astype(int)

    return X, y

def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy dataset tại: {DATA_PATH}")
    print("Đang tải bộ dữ liệu PhiUSIIL...")
    df = pd.read_csv(DATA_PATH)
    X, y = select_lexical_features(df)

    #tính số cột HTML bị loại
    known_non_html = (
        set(ID_AND_LABEL_COLS)
        | set(LEAKAGE_AND_RAW_COLS)
        | set (REDUNDANT_LEXICAL_COLS)
        | set (SELECTED_LEXICAL_FEATURES)
    )
    html_cols = [c for c in df.columns if c not in known_non_html]

    print("\n================ BÁO CÁO LỌC ĐẶC TRƯNG (FEATURE SELECTION) ================")
    print(f"Tổng số cột ban đầu trong Dataset             : {df.shape[1]} cột")
    print(f"  [-] Cột định danh & nhãn (ID/Label) : -{len(ID_AND_LABEL_COLS)} cột")
    print(f"  [-] Cột nội dung HTML/DOM ngoại lai : -{len(html_cols)} cột")
    print(f"  [-] Cột chữ thô & Rò rỉ nhãn        : -{len(LEAKAGE_AND_RAW_COLS)} cột")
    print(f"  [-] Cột từ vựng trùng lặp/đa cộng tuyến: -{len(REDUNDANT_LEXICAL_COLS)} cột")
    print("---------------------------------------------------------------------------")
    print(f"  [=] SỐ ĐẶC TRƯNG TỪ VỰNG CỐT LÕI ĐƯỢC CHỌN   : {X.shape[1]} cột")
    print("===========================================================================")

    # tạo bảng so sánh giá trị trung bình của từng đặc trưng giữa nhóm phishing và legitimate
    legit_mean = X[y == 0].mean()
    phish_mean = X[y == 1].mean()

    report_df = pd.DataFrame({
        "Ý nghĩa": [FEATURE_DESCRIPTIONS[col] for col in SELECTED_LEXICAL_FEATURES],
        "Avg Legit (0)": [round(legit_mean[col], 4) for col in SELECTED_LEXICAL_FEATURES],
        "Avg Phish (1)": [round(phish_mean[col], 4) for col in SELECTED_LEXICAL_FEATURES],
        "Std toàn tập": [round(X[col].std(), 4) for col in SELECTED_LEXICAL_FEATURES],
    }, index = SELECTED_LEXICAL_FEATURES)

    print("\n--- Bảng thống kê 10 đặc trưng từ vựng (So sánh trung bình Legitimate vs Phishing) ---")
    print(report_df.to_string())

if __name__ == "__main__":
    main()