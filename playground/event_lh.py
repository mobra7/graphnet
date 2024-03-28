import sqlite3
import random

path = '/mnt/scratch/rasmus_orsoe/databases/dev_northern_tracks_muon_labels_v3/dev_northern_tracks_muon_labels_v3_part_1.db'
# Connect to the SQLite database
conn = sqlite3.connect(path)
cursor = conn.cursor()

# Fetch the headers of the 'truth' table
cursor.execute("PRAGMA table_info(truth)")
headers = [header[1] for header in cursor.fetchall()]

# Select 1 random row from the 'truth' table
cursor.execute("SELECT * FROM truth ORDER BY RANDOM() LIMIT 1")

# Fetch and print the selected row
random_row = cursor.fetchone()
print("Random Row:")
for header, value in zip(headers, random_row):
    print(f"{header}: {value}")

# Don't forget to close the cursor and connection
cursor.close()
conn.close()