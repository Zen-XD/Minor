import sqlite3

DATABASE = 'users.db'
username = "testuser"
email = "test@example.com"
password = "12345678"

try:
    dbms = sqlite3.connect(DATABASE)
    cursor = dbms.cursor()
    cursor.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)', 
                (username, email, password))
    dbms.commit()
    dbms.close()
    print("Data inserted successfully.")
except sqlite3.IntegrityError as e:
    print(f"Database error: {e}")

