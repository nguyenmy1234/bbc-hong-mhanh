"""
BBC News Classification - All-in-One Pipeline
Includes: EDA Analysis, ML Classifiers, Benchmark & Deep Learning LSTM
Tác giả: Hạnh
Dữ liệu: dataset.csv
"""

import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def load_dataset():
    path = 'dataset.csv' if os.path.exists('dataset.csv') else 'dataset'
    if not os.path.exists(path):
        path = 'https://raw.githubusercontent.com/mdsohaibuddin/BBC-News-Classification/master/bbc-text.csv'
    print(f"📥 Đang đọc tập dữ liệu từ: {path}")
    return pd.read_csv(path)

def run_eda(df):
    print("\n--- 1. KHÁM PHÁ DỮ LIỆU (EDA) ---")
    print(f"✓ Tổng số mẫu: {len(df)}")
    print("✓ Phân bố danh mục:\n", df['category'].value_counts())

def run_ml(df):
    print("\n--- 2. HUẤN LUYỆN MACHINE LEARNING (TF-IDF) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['category'], test_size=0.2, random_state=42, stratify=df['category']
    )
    vec = TfidfVectorizer(max_features=5000, stop_words='english')
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    print(f"✅ Accuracy Logistic Regression: {accuracy_score(y_test, preds)*100:.2f}%")

def run_lstm(df):
    print("\n--- 3. HUẤN LUYỆN DEEP LEARNING (LSTM) ---")
    tokenizer = Tokenizer(num_words=10000, lower=True)
    tokenizer.fit_on_texts(df['text'])
    X = pad_sequences(tokenizer.texts_to_sequences(df['text']), maxlen=500)
    
    categories = sorted(df['category'].unique())
    label_map = {cat: i for i, cat in enumerate(categories)}
    Y = to_categorical(df['category'].map(label_map))

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = Sequential([
        Embedding(10000, 128, input_length=500),
        SpatialDropout1D(0.2),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dense(len(categories), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=3, batch_size=64, verbose=1)
    
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Accuracy LSTM: {acc*100:.2f}%")

if __name__ == '__main__':
    df = load_dataset()
    run_eda(df)
    run_ml(df)
    run_lstm(df)
