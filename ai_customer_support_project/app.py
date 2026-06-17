# app.py
from flask import Flask, render_template, request
import joblib
import sqlite3
import os
import nltk
from nltk.tokenize import word_tokenize

app = Flask(__name__)
DB_PATH = os.path.join('data', 'reviews_history.db')

# Попытка загрузить обученные ML-артефакты
try:
    model = joblib.load('review_classifier.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    print("WARNING: Модели не найдены! Сначала запустите train_pipeline.py")

# Шаблоны ответов поддержки по бизнес-категориям
RESPONSE_TEMPLATES = {
    'DEFECT': "We are deeply sorry that you received a defective product. Please contact our support team for an immediate replacement.",
    'DELIVERY': "We apologize for the delivery delay. Our logistics team is investigating the issue and will update you shortly.",
    'REFUND': "We understand your request. A return label and refund instructions have been sent to your email.",
    'QUALITY': "Thank you for your feedback regarding product quality. We have forwarded this to our QA department.",
    'POSITIVE': "Thank you so much for your excellent review! We are thrilled that you enjoyed the product.",
    'OTHER': "Thank you for reaching out. A customer support representative will review your request within 24 hours."
}

def rule_based_category(text):
    """Определение категории по ключевым словам."""
    text_lower = text.lower()
    if any(w in text_lower for w in ['damage', 'broken', 'defect', 'cracked', 'scratch', 'faulty']):
        return 'DEFECT'
    elif any(w in text_lower for w in ['delivery', 'delay', 'shipped', 'shipping', 'arrive', 'late']):
        return 'DELIVERY'
    elif any(w in text_lower for w in ['refund', 'return', 'money back', 'charge', 'refunded']):
        return 'REFUND'
    elif any(w in text_lower for w in ['quality', 'material', 'fabric', 'build', 'screen', 'display']):
        return 'QUALITY'
    elif any(w in text_lower for w in ['excellent', 'good', 'love', 'perfect', 'great', 'recommended', 'best']):
        return 'POSITIVE'
    return 'OTHER'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    
    if request.method == 'POST':
        user_review = request.form.get('review_text', '').strip()
        
        if user_review:
            # 1. Предсказание тональности с помощью вашей ML-модели
            if model_loaded:
                try:
                    cleaned_text = " ".join([w.lower() for w in word_tokenize(user_review) if w.isalnum()])
                    vec_text = vectorizer.transform([cleaned_text])
                    ml_sentiment = model.predict(vec_text)[0] # 'positive' или 'negative'
                    sentiment_badge = "🟢 Positive" if ml_sentiment == 'positive' else "🔴 Negative"
                except Exception as e:
                    sentiment_badge = "🟡 Neutral (Error during ML inference)"
            else:
                sentiment_badge = "🟡 Neutral (Model not trained yet)"
            
            # 2. Определение категории
            category = rule_based_category(user_review)
            if sentiment_badge == "🟢 Positive" and category == "OTHER":
                category = "POSITIVE"
                
            # 3. Подбор ответа
            support_response = RESPONSE_TEMPLATES.get(category, RESPONSE_TEMPLATES['OTHER'])
            
            # 4. Сохранение в SQLite
            try:
                conn = get_db_connection()
                conn.execute('''
                    INSERT INTO review_logs (review_text, predicted_category, sentiment, response_text)
                    VALUES (?, ?, ?, ?)
                ''', (user_review, category, sentiment_badge, support_response))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Database insertion error: {e}")
            
            prediction_result = {
                'review': user_review,
                'category': category,
                'sentiment': sentiment_badge,
                'response': support_response
            }
            
    # Получение истории и аналитики
    history = []
    stats = []
    if os.path.exists(DB_PATH):
        try:
            conn = get_db_connection()
            history = conn.execute('SELECT * FROM review_logs ORDER BY timestamp DESC LIMIT 5').fetchall()
            stats = conn.execute('SELECT predicted_category, COUNT(*) as count FROM review_logs GROUP BY predicted_category').fetchall()
            conn.close()
        except Exception as e:
            print(f"Database read error: {e}")
            
    return render_template('index.html', result=prediction_result, history=history, stats=stats)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
