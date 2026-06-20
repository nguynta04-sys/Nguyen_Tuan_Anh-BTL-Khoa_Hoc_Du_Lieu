import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================================
# BƯỚC 1: KHỞI TẠO BỘ DỮ LIỆU CHUỖI THỜI GIAN GIẢ LẬP (2015 - 2025)
# =====================================================================
np.random.seed(42)
# Khởi tạo chuỗi thời gian theo tháng (MS: Month Start) từ năm 2015 đến 2025
dates = pd.date_range(start='2015-01-01', end='2025-12-01', freq='MS')
n_samples = len(dates)

# Giả lập xu hướng tăng trưởng tuyến tính (Trend) của Xuất khẩu và Nhập khẩu (Tỷ USD)
# Giả lập nền kinh tế Việt Nam có xu hướng tăng trưởng bền vững và duy trì xuất siêu
trend_export = np.linspace(15, 38, n_samples) + np.random.normal(0, 1.2, n_samples)
trend_import = np.linspace(14, 34, n_samples) + np.random.normal(0, 1.0, n_samples)

# Thêm yếu tố chu kỳ/mùa vụ ngắn hạn (Seasonality) bằng hàm Sin (cuối năm tăng, đầu năm giảm)
seasonality = np.sin(2 * np.pi * dates.month / 12) * 1.8
export_values = trend_export + seasonality
import_values = trend_import + seasonality * 0.85 # Nhập khẩu biến động mùa vụ ít hơn

# Gom vào DataFrame đại diện cho dữ liệu vĩ mô thô thu thập được
df = pd.DataFrame({
    'Ngay': dates,
    'Xuat_Khau': export_values,
    'Nhap_Khau': import_values
})

# CHỦ ĐỘNG GIEO LỖI KHUYẾT (NaN) ĐỂ DEMO PHẦN TIỀN XỬ LÝ (Lọc ngẫu nhiên 3% số hàng bị mất dữ liệu)
df.loc[df.sample(frac=0.03, random_state=10).index, 'Xuat_Khau'] = np.nan
df.loc[df.sample(frac=0.03, random_state=20).index, 'Nhap_Khau'] = np.nan

print("--- 5 DÒNG DỮ LIỆU THÔ BAN ĐẦU (CÓ CHỨA NaN NẾU RƠI VÀO Ô LỖI) ---")
print(df.head(6))


# =====================================================================
# BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU VÀ FEATURE ENGINEERING
# =====================================================================
# 1. Xử lý dữ liệu khuyết bằng phương pháp nội suy tuyến tính (Linear Interpolation) phù hợp chuỗi thời gian
df['Xuat_Khau'] = df['Xuat_Khau'].interpolate(method='linear')
df['Nhap_Khau'] = df['Nhap_Khau'].interpolate(method='linear')

# 2. Tạo đặc trưng mới: Tính toán Cán cân thương mại = Xuất khẩu - Nhập khẩu
df['Can_Can_Thuong_Mai'] = df['Xuat_Khau'] - df['Nhap_Khau']

# 3. Tạo biến độc lập Time_Index đại diện cho thời gian lũy tiến (0, 1, 2, 3...) phục vụ Linear Regression
df['Time_Index'] = np.arange(len(df))


# =====================================================================
# BƯỚC 3: PHÂN TÁCH TẬP DỮ LIỆU HUẤN LUYỆN VÀ KIỂM THỬ (TRAIN/TEST SPLIT)
# =====================================================================
# Phân chia biến độc lập (X) và biến phụ thuộc/biến mục tiêu cần dự báo (y)
X = df[['Time_Index']]
y = df['Can_Can_Thuong_Mai']

# LƯU Ý SỐNG CÒN: Không chia ngẫu nhiên (train_test_split mặc định) đối với chuỗi thời gian!
# Ta phân tách 80% dữ liệu đầu để huấn luyện (Train) và 20% dữ liệu cuối để kiểm thử (Test)
train_size = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]


# =====================================================================
# BƯỚC 4: CHUẨN HÓA DỮ LIỆU (FEATURE SCALING)
# =====================================================================
# Đưa biến độc lập thời gian về phân phối chuẩn để mô hình hồi quy hội tụ ổn định, chính xác
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Dùng lại tham số scaler của tập Train


# =====================================================================
# BƯỚC 5: HUẤN LUYỆN MÔ HÌNH HỒI QUY TUYẾN TÍNH (LINEAR REGRESSION)
# =====================================================================
model = LinearRegression()
model.fit(X_train_scaled, y_train)


# =====================================================================
# BƯỚC 6: ĐÁNH GIÁ CHẤT LƯỢNG MÔ HÌNH TRÊN TẬP KIỂM THỬ (TEST SET)
# =====================================================================
y_pred = model.predict(X_test_scaled)

# Tính toán các chỉ số đo lường hiệu năng cốt lõi bắt buộc trong tiểu luận
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("KẾT QUẢ ĐÁNH GIÁ HIỆU SUẤT MÔ HÌNH")
print("="*50)
print(f"Hệ số xác định (R2 Score): {r2:.4f}")
print(f"Sai số Căn trung bình bình phương (RMSE): {rmse:.4f} Tỷ USD")


# =====================================================================
# BƯỚC 7: TIẾN HÀNH DỰ BÁO TƯƠNG LAI CHO CẢ NĂM 2026 (12 THÁNG)
# =====================================================================
# Tạo mốc thời gian tương lai cho 12 tháng năm 2026
future_months = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')

# Tiếp tục lũy tiến chỉ số Time_Index cho 12 tháng tiếp theo
future_time_index = np.arange(len(df), len(df) + len(future_months)).reshape(-1, 1)

# Chuẩn hóa biến thời gian tương lai bằng scaler gốc
future_time_index_scaled = scaler.transform(future_time_index)

# Chạy mô hình dự báo
future_predictions = model.predict(future_time_index_scaled)

# Tổng hợp kết quả dự báo ra bảng dữ liệu sạch
df_future = pd.DataFrame({
    'Tháng/Năm Tương Lai': future_months.strftime('%m/%Y'),
    'Dự Báo Cán Cân Thương Mại (Tỷ USD)': np.round(future_predictions, 3),
    'Xu Hướng': ['XUẤT SIÊU' if val > 0 else 'NHẬP SIÊU' for val in future_predictions]
})

print("\n" + "="*50)
print("BẢNG DỰ BÁO CÁN CÂN THƯƠNG MẠI VIỆT NAM NĂM 2026")
print("="*50)
print(df_future.to_string(index=False))