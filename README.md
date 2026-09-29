# 📰 BBC News Text Classification & AI Analytics

Báo cáo Bài tập lớn môn: **Nền tảng Lập trình cho Phân tích và Trực quan Dữ liệu**  
- **Thực hiện bởi:** Hạnh  
- **Tập dữ liệu:** `dataset.csv` (2,225 bài báo BBC)  

🌐 **[XEM TRANG WEB DASHBOARD TƯƠNG TÁC TẠI ĐÂY](https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/)**

---

## 📌 Tổng quan Cấu trúc Repository

1. **`index.html`**: Trang Web View tương tác (Hiển thị biểu đồ EDA, Bảng so sánh kết quả và Công cụ AI dự đoán trực tiếp).
2. **`dataset.csv`**: File dữ liệu gốc chứa 2,225 bài báo phân làm 5 danh mục.
3. **`main_pipeline.py`**: File mã nguồn Python chạy toàn bộ quy trình EDA, Machine Learning và Deep Learning LSTM.

---

## 📊 Kết quả Huấn luyện Mô hình

| Thuật toán | Loại mô hình | Accuracy (%) | Thời gian huấn luyện |
| :--- | :--- | :---: | :---: |
| **Voting Ensemble** | Machine Learning | **98.20%** | 4.20s |
| **Logistic Regression** | Machine Learning | 97.60% | 0.42s |
| **LSTM Network** | Deep Learning | **97.30%** | 15.00s |
| **MLP Classifier** | Neural Network | 97.30% | 3.80s |
| **Multinomial Naive Bayes** | Machine Learning | 96.10% | **0.05s** |
