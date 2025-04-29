import sqlite3
import os

DB_PATH = "data/course_content.db"  


conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()


cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print(" Tables in database:", [table[0] for table in tables])


for table in tables:
    table_name = table[0]
    print(f"\ Data in {table_name}:")
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    for row in rows:
        print(row)

conn.close()



import os

DB_PATH = "data/course_content.db"  

if os.path.exists(DB_PATH):
    print(f" Python is using: {os.path.abspath(DB_PATH)}")
    print(f"File size: {os.path.getsize(DB_PATH)} bytes")
else:
    print(" Database file NOT found")

