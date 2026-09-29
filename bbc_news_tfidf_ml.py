"""
BBC News Text Classification - TF-IDF + Traditional Machine Learning
Created by: Hạnh
Data Source: dataset.csv
"""

import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

def load_data():
    """Tải dữ liệu từ file dataset.csv"""
    print("="*70)
    print("📥 LOADING DATASET (dataset.csv)")
    print("="*70)
    
    # Đọc trực tiếp file dataset.csv
    try:
        df = pd.read_csv('dataset.csv')
    except FileNotFoundError:
        # Dự phòng nếu file ở dạng link
        url = 'https://raw.githubusercontent.com/mdsohaibuddin/BBC-News-Classification/master/bbc-text.csv'
        df = pd.read_csv(url)
        
    print(f"✓ Total samples: {len(df):,}")
    print(f"✓ Categories: {sorted(df['category'].unique().tolist())}\n")
    return df

def main():
    df = load_data()
    
    # Chia tập dữ liệu Train / Test
    X_train_text, X_test_text, y_train_raw, y_test_raw = train_test_split(
        df['text'], df['category'], test_size=0.2, random_state=42, stratify=df['category']
    )
    
    # Trích xuất đặc trưng TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    
    # Mã hóa nhãn
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)
    
    # Huấn luyện mô hình Logistic Regression đại diện
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy Logistic Regression: {acc*100:.2f}%")

if __name__ == '__main__':
    main()
