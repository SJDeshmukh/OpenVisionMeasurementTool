import sqlite3

def extract_data_from_db(db_file):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    
    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()
    
    # Fetch all rows from the users table
    cursor.execute("SELECT * FROM measurements;")
    rows = cursor.fetchall()
    
    if not rows:
        print("No data found in the users table.")
        return
    
    # Get column names from the cursor description
    column_names = [description[0] for description in cursor.description]
    
    # Print header
    print(f"Data extracted from users table:")
    print(" | ".join(column_names))  # Print column names as header
    
    # Print each row of data
    for row in rows:
        print(" | ".join(str(value) for value in row))  # Print row values
    
    # Close the connection
    conn.close()

# Example usage:
extract_data_from_db('measurement_tool.db')