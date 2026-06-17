# train_pipeline.py
import os
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Скачивание необходимых ресурсов NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def train_and_save_model():
    print("Запуск пайплайна обучения...")
    
    # Имитация датасета (на случай если Dataset (3).csv отсутствует при первом запуске)
    # Если файл есть, мы можем загрузить его
    csv_path = 'data/Dataset (3).csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Приведение колонок к нужному формату
        df.columns = [c.lower().strip() for c in df.columns]
    else:
        # Синтетические данные для сборки проекта
        data = {
            'review': [
                "The phone arrived damaged and the screen is cracked broken defect.",
                "The delivery was delayed for a week, shipping took forever.",
                "I would like to return this product and get a refund money back.",
                "Excellent quality, highly recommended to everyone! Amazing build.",
                "The item is okay, nothing special but works fine.",
                "Terrible service, the package was completely broken.",
                "Shipping was late. Horrible delivery experience.",
                "Can I get a refund for this damaged item?",
                "Amazing build quality and fast shipping!",
                "It's an average product, could be better."
            ],
            'sentiment': ['negative', 'negative', 'negative', 'positive', 'positive', 
                          'negative', 'negative', 'negative', 'positive', 'positive']
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"Создан демонстрационный датасет в {csv_path}")

    # Предобработка текста
    stop_words = set(stopwords.words('english'))
    
    def clean_text(text):
        if not isinstance(text, str): return ""
        tokens = word_tokenize(text.lower())
        return " ".join([w for w in tokens if w.isalnum() and w not in stop_words])

    df['cleaned_text'] = df['review'].apply(clean_text)
    
    # Векторизация
    vectorizer = TfidfVectorizer(max_features=2500)
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['sentiment']
    
    # Обучение
    model = LogisticRegression(C=1.0, solver='lbfgs')
    model.fit(X, y)
    
    # Сохранение артефактов
    joblib.dump(model, 'review_classifier.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("Модель (review_classifier.pkl) и Векторизатор (tfidf_vectorizer.pkl) успешно сохранены!")

if __name__ == '__main__':
    train_and_save_model()
