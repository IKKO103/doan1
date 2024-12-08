import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import yfinance as yf
import streamlit as st

# Bước 1: Tải dữ liệu lịch sử giá cổ phiếu
def load_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Change'] = data['Close'].pct_change()  # Tính phần trăm thay đổi giá đóng cửa
    data['Trend'] = (data['Change'] > 0).astype(int)  # Xu hướng: 1 (Tăng), 0 (Giảm)
    data = data.dropna()  # Loại bỏ giá trị NaN
    return data

# Bước 2: Tiền xử lý dữ liệu
def preprocess_data(data):
    # Sử dụng các cột đặc trưng: Giá mở cửa, giá đóng cửa, khối lượng giao dịch
    features = data[['Open', 'Close', 'Volume']]
    target = data['Trend']  # Mục tiêu: Xu hướng
    return train_test_split(features, target, test_size=0.2, random_state=42)

# Bước 3: Xây dựng mô hình cây quyết định
def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42, max_depth=5)  # Cây phân lớp
    model.fit(X_train, y_train)
    return model

# Bước 4: Đánh giá mô hình
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, classification_report(y_test, y_pred)

# Bước 5: Ứng dụng chương trình với Streamlit
if __name__ == "__main__":
    st.title("Dự đoán Xu Hướng Giá Cổ Phiếu")

    # Nhập thông tin từ người dùng
    ticker = st.text_input("Nhập mã cổ phiếu (ví dụ: AAPL):", "AAPL")
    start_date = st.date_input("Chọn ngày bắt đầu:", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("Chọn ngày kết thúc:", value=pd.to_datetime("2023-01-01"))

    if st.button("Tải dữ liệu và Dự đoán"):
        # Tải và xử lý dữ liệu
        st.write("Đang tải dữ liệu...")
        data = load_stock_data(ticker, start_date, end_date)
        if data.empty:
            st.write("Không có dữ liệu cho mã cổ phiếu này trong khoảng thời gian được chọn.")
        else:
            X_train, X_test, y_train, y_test = preprocess_data(data)

            # Huấn luyện mô hình
            model = train_decision_tree(X_train, y_train)

            # Đánh giá mô hình
            accuracy, report = evaluate_model(model, X_test, y_test)
            st.write(f"Độ chính xác của mô hình: {accuracy:.2f}")
            st.text(report)

            # Dự đoán xu hướng ngày tiếp theo
            st.write("Dự đoán xu hướng ngày tiếp theo:")
            open_price = st.number_input("Giá mở cửa:", value=150.0)
            close_price = st.number_input("Giá đóng cửa:", value=155.0)
            volume = st.number_input("Khối lượng giao dịch:", value=50000000.0)
            new_data = np.array([[open_price, close_price, volume]])
            prediction = model.predict(new_data)
            trend = "Tăng" if prediction[0] == 1 else "Giảm"
            st.write(f"Xu hướng dự đoán: {trend}")
