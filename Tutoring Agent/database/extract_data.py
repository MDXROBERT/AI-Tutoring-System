import sqlite3
import json
import os

DB_PATH = "../data/course_content.db"
TRAINING_FILE = "../data/training_data.json"
QUESTION_FILE = "../data/training_questions.json"


def extract_course_data():
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT topic, subtopic, summary, detailed_content FROM course_content")
    rows = cursor.fetchall()
    conn.close()

    data = [{"topic": row[0], "subtopic": row[1], "summary": row[2], "detailed_content": row[3]} for row in rows]

    os.makedirs("data", exist_ok=True)
    with open(TRAINING_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f" Course data extracted to {TRAINING_FILE}")

def extract_questions():
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT topic, question FROM questions")
    rows = cursor.fetchall()
    conn.close()

    data = [{"topic": row[0], "question": row[1]} for row in rows]

    with open(QUESTION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f" Questions extracted to {QUESTION_FILE}")

if __name__ == "__main__":
    extract_course_data()
    extract_questions()
