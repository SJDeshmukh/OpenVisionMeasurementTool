import sqlite3

def get_db_connection(db_file):
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row  # To get dictionary-like row results
    return conn

def execute_delete_all_query(db_file):
    try:
        conn = get_db_connection(db_file)
        cursor = conn.cursor()
        cursor.execute("DELETE from measurements;")  # Delete all rows in the 'users' table
        conn.commit()  # Commit the transaction
        print("All data deleted successfully!")
    except Exception as e:
        print(f"Error executing DELETE query: {e}")
    finally:
        conn.close()

# Example usage:
db_file = 'measurement_tool.db'
execute_delete_all_query(db_file)