"""
BBC News Text Classification using LSTM & Deep Learning
Tác giả: Hạnh
Dữ liệu sử dụng: dataset.csv (hoặc bbc-text.csv đổi tên)
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# NLTK cho tiền xử lý văn bản
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# TensorFlow / Keras (API TensorFlow 2.x mới nhất)
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Đánh giá mô hình
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Tải NLTK Stopwords
nltk.download('stopwords', quiet=True)

# ============================================================================
# 1. TẢI VÀ KIỂM TRA DỮ LIỆU
# ============================================================================
print("="*70)
print("📥 DẠNG DỮ LIỆU BÀI BÁO BBC (DATASET)")
print("="*70)

# Kiểm tra file dataset.csv hoặc dataset
dataset_path = 'dataset.csv'
if not os.path.exists(dataset_path):
    if os.path.exists('dataset'):
        dataset_path = 'dataset'
    elif os.path.exists('bbc-text.csv'):
        dataset_path = 'bbc-text.csv'
    else:
        # Nếu chưa có file tại máy thì tải tự động
        dataset_path = 'https://raw.githubusercontent.com/mdsohaibuddin/BBC-News-Classification/master/bbc-text.csv'

print(f"✓ Đang đọc dữ liệu từ: {dataset_path}")
df = pd.read_csv(dataset_path)

print(f"✓ Kích thước dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột")
print("\n--- 5 Dòng đầu tiên của Dataset ---")
print(df.head())

print("\n--- Phân bố các danh mục bài báo ---")
print(df['category'].value_counts())


# ============================================================================
# 2. TIỀN XỬ LÝ VĂN BẢN (TEXT CLEANING)
# ============================================================================
print("\n" + "="*70)
print("🧹 TIỀN XỬ LÝ VĂN BẢN (CLEANING & STEMMING)")
print("="*70)

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    # Loại bỏ ký tự đặc biệt & số, chỉ giữ lại chữ cái A-Z
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    # Loại bỏ stopwords & Chuẩn hóa từ gốc (Stemming)
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

print("✓ Đang xử lý các đoạn văn bản...")
corpus = [clean_text(text) for text in df['text']]
print(f"✓ Hoàn tất làm sạch {len(corpus)} bài viết.")


# ============================================================================
# 3. TOKENIZATION, PADDING & MÃ HÓA NHÃN
# ============================================================================
print("\n" + "="*70)
print("🔢 TOKENIZATION & MA TRẬN HÓA DỮ LIỆU")
print("="*70)

# Cấu hình Tokenizer
MAX_WORDS = 10000        # Lấy tối đa 10,000 từ phổ biến nhất
MAX_LEN = 500            # Độ dài cố định của mỗi chuỗi đầu vào
EMBEDDING_DIM = 128      # Chiều của không gian nhúng Embedding

tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True, split=' ')
tokenizer.fit_on_texts(corpus)

X = tokenizer.texts_to_sequences(corpus)
X = pad_sequences(X, maxlen=MAX_LEN)

print(f"✓ Dạng Ma trận Đầu vào X (Shape): {X.shape}")

# Mã hóa nhãn Categorical (Target mapping)
categories = sorted(df['category'].unique())
label_map = {cat: i for i, cat in enumerate(categories)}
df['target'] = df['category'].map(label_map)

# One-hot encoding nhãn Y
Y = to_categorical(df['target'], num_classes=len(categories))
print(f"✓ Dạng Ma trận Đầu ra Y (Shape): {Y.shape}")
print(f"✓ Danh sách Nhãn: {label_map}")


# ============================================================================
# 4. CHIA TẬP DỮ LIỆU (TRAIN / TEST)
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42, stratify=df['target']
)

print(f"✓ Tập huấn luyện (Train): {X_train.shape[0]} mẫu")
print(f"✓ Tập kiểm thử (Test):     {X_test.shape[0]} mẫu")


# ============================================================================
# 5. XÂY DỰNG MÔ HÌNH DEEP LEARNING (LSTM)
# ============================================================================
print("\n" + "="*70)
print("🏗️ XÂY DỰNG MÔ HÌNH LSTM")
print("="*70)

vocab_size = min(len(tokenizer.word_index) + 1, MAX_WORDS)

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    SpatialDropout1D(0.2),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(categories), activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()


# ============================================================================
# 6. HUẤN LUYỆN MÔ HÌNH (TRAINING)
# ============================================================================
print("\n" + "="*70)
print("🚀 ĐANG HUẤN LUYỆN MÔ HÌNH...")
print("="*70)

BATCH_SIZE = 64
EPOCHS = 10

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    verbose=1
)


# ============================================================================
# 7. ĐÁNH GIÁ VÀ TRỰC QUAN HÓA KẾT QUẢ
# ============================================================================
print("\n" + "="*70)
print("📊 ĐÁNH GIÁ HIỆU NĂNG MÔ HÌNH")
print("="*70)

# Dự đoán trên tập Test
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

acc = accuracy_score(y_true, y_pred)
print(f"\n🎯 Độ chính xác tổng thể (Accuracy): {acc*100:.2f}%\n")

print("--- Báo cáo Chi tiết (Classification Report) ---")
print(classification_report(y_true, y_pred, target_names=categories))

# Vẽ biểu đồ Accuracy & Loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
plt.title('Mô hình Accuracy qua các Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
plt.title('Mô hình Loss qua các Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Vẽ Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.title('Confusion Matrix - LSTM BBC News')
plt.xlabel('Dự đoán (Predicted)')
plt.ylabel('Thực tế (Actual)')
plt.show()
