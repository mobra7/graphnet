import sqlite3


database = '/mnt/scratch/rasmus_orsoe/databases/dev_northern_tracks_muon_labels_v3/dev_northern_tracks_muon_labels_v3_part_1.db'
pulsemap = 'InIcePulses'
event_nos = [(879,1),(847,0),(850,1),(850,0),(879,0),(849,1)]

# Connect to the database
conn = sqlite3.connect(database)

# Build the SQL query to get counts of rows for each event number
query = f"SELECT event_no, COUNT(*) FROM {pulsemap} WHERE event_no IN ({','.join(str(event[0]) for event in event_nos)}) GROUP BY event_no HAVING COUNT(*) BETWEEN ? AND ?"

# Define the range of counts
min_count = 60
max_count = 200

# Execute the query and fetch all rows
result = conn.execute(query, (min_count, max_count)).fetchall()

# Filter event numbers based on the count range, retaining other values
filtered_event_nos = [(event_no, other_value) for event_no, other_value in event_nos if any(event_no == row[0] for row in result)]

print(filtered_event_nos)
