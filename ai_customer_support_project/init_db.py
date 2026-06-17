# init_db.py
import sqlite3
import os

def init_database():
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect('data/reviews_history.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS review_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text TEXT NOT NULL,
            predicted_category TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            response_text TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("База данных 'data/reviews_history.db' успешно инициализирована!")

if __name__ == '__main__':
    init_database()
