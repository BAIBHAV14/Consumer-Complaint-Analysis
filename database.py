import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('user_details.db')
cursor = conn.cursor()

# Create a table to store user details
cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_details (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_name TEXT,
        issue TEXT,
        company_public_response TEXT,
        company_name TEXT,
        tags TEXT,
        submission_location TEXT,
        company_response TEXT,
        timely_response TEXT
    )
''')
conn.commit()

# Function to insert user details into the database
def insert_user_details(product_name, issue, company_public_response, company_name, tags, submission_location, company_response, timely_response):
    cursor.execute('''
        INSERT INTO user_details (
            product_name,
            issue,
            company_public_response,
            company_name,
            tags,
            submission_location,
            company_response,
            timely_response
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (product_name, issue, company_public_response, company_name, tags, submission_location, company_response, timely_response))
    conn.commit()

# Close the connection
def close_connection():
    conn.close()


def get_user_details():
    conn = sqlite3.connect('user_details.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM user_details
    ''')

    user_details = cursor.fetchall()

    conn.close()

    return user_details