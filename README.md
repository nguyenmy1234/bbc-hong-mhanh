# 📰 BBC News Text Classification & EDA - Full Project

Báo cáo Bài tập lớn: **Phân tích và Phân loại Tin tức BBC (BBC News Dataset)**  
- **Thực hiện bởi:** Hạnh  
- **Lĩnh vực:** Xử lý Ngôn ngữ Tự nhiên (NLP) & Machine Learning  

🌐 **[XEM GIAO DIỆN WEB DỰ ĐOÁN INTERACTIVE TẠI ĐÂY](https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/)**

---

## 📌 Cấu trúc Toàn bộ Bài tập trong Repository

1. **`index.html`**: Giao diện Web Dashboard tổng hợp toàn bộ biểu đồ EDA, Bảng so sánh kết quả và Công cụ AI dự đoán trực tiếp.
2. **`eda_category_distribution.py`**: Mã nguồn phân tích phân bố bài báo theo từng chủ đề.
3. **`eda_vocabulary_richness.py`**: Mã nguồn phân tích độ phong phú từ vựng (Vocabulary Richness - TTR).
4. **`eda_character_count_distribution.py`**: Mã nguồn phân tích độ dài bài báo và từ vựng.
5. **`bbc_news_tfidf_ml.py`**: Mã nguồn huấn luyện các mô hình Machine Learning chính (Logistic Regression, Random Forest, MLP, Voting Ensemble...).
6. **`bbc_pipeline_comparison.py`**: Mã nguồn thử nghiệm Benchmark 240 Pipelines khác nhau.

---

## 📊 1. Khám phá & Trực quan hóa Dữ liệu (EDA)

Tập dữ liệu bao gồm **2,225 bài báo BBC** chia làm 5 chủ đề: *Business, Entertainment, Politics, Sport, Tech*.
- **Phân bố chủ đề:** Số lượng bài viết giữa các nhóm tương đối đồng đều (~380 - 510 bài/chủ đề).
- **Độ dài bài viết:** Độ dài trung bình dao động khoảng **2,200 ký tự/bài báo**, độ dài từ trung bình khoảng **5 ký tự**.
- **Phong phú từ vựng:** Danh mục *Tech* và *Politics* có tỷ lệ từ vựng đơn nhất (Unique Words) cao nhất.

---

## 🚀 2. Kết quả Huấn luyện Machine Learning

Sử dụng phương pháp trích xuất đặc trưng **TF-IDF (5,000 max features, n-grams 1-2)** kết hợp loại bỏ English Stopwords:

| Mô hình | Accuracy (%) | Precision | Recall | F1-Score | Thời gian |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Voting Ensemble (Top 3)** | **98.20%** | **0.9822** | **0.9820** | **0.9820** | 4.20s |
| **Logistic Regression** | 97.60% | 0.9762 | 0.9760 | 0.9759 | 0.42s |
| **MLP Classifier** | 97.30% | 0.9735 | 0.9730 | 0.9731 | 3.80s |
| **Linear SVC** | 97.00% | 0.9701 | 0.9700 | 0.9698 | 0.18s |
| **Multinomial Naive Bayes** | 96.10% | 0.9620 | 0.9610 | 0.9608 | **0.05s** |

---

## 💻 Cách chạy các file mã nguồn Python

### 1. Cài đặt thư viện
```bash
pip install pandas numpy scikit-learn plotly xgboost
