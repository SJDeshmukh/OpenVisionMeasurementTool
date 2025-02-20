import sqlite3

def create_db_and_table():
    # Connect to SQLite database (this creates the database file if it doesn't exist)
    conn = sqlite3.connect('measurement_tool.db')
    
    # Create a cursor object
    cursor = conn.cursor()

    # Create the table for storing image paths and JSON data
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS measurements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT NOT NULL,
        json_data TEXT NOT NULL
    )
    ''')

    # Commit changes and close the connection
    conn.commit()
    cursor.close()
    conn.close()

    print("Database and table created successfully!")

# Call the function to create the database and table
create_db_and_table()
