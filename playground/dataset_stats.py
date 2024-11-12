import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path = "/scratch/users/allorana/northern_sqlite/files_no_hlc/dev_northern_tracks_full_part_1.db"

conn = sqlite3.connect(path)
indices = pd.read_sql_query(
        f"SELECT event_no FROM truth", conn
    )
# Filter based on pulse count
query = f"SELECT event_no FROM InIcePulses WHERE event_no IN ({','.join(map(str, indices))}) GROUP BY event_no HAVING COUNT(*) BETWEEN ? AND ?"

min_count = 1
max_count = 1000

print('executing first query')
event_nos = [event_no for event_no, in conn.execute(query, (min_count, max_count)).fetchall()]

print('executing second query')
event_numbers_str = ','.join(map(str, event_nos))
query = f"SELECT azimuth, zenith, energy  FROM truth WHERE event_no IN ({event_numbers_str});"
df = pd.read_sql_query(query, conn)


plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.histplot(df['azimuth'], bins=30)
plt.xlabel('Azimuth [rad]')


plt.subplot(1, 3, 2)
sns.histplot(np.cos(df['zenith']), bins=30)
plt.yscale('log')
plt.xlabel('cos(Zenith) [cos(rad)]')


plt.subplot(1, 3, 3)
sns.histplot(df['energy'], bins=30, log_scale=True)
plt.yscale('log')
plt.xlabel('Energy [GeV]')
plt.tight_layout()

plt.savefig('./dataset_stats/combined_plots.pdf')

conn = sqlite3.connect(path)
cursor = conn.cursor()

print('executing third query')
query = """
SELECT event_no, COUNT(*) as pulse_count
FROM InIcePulses
GROUP BY event_no;
"""
cursor.execute(query)


results = cursor.fetchall()

# Close the connection
conn.close()

# Step 3: Extract pulse counts for plotting
pulse_counts = [row[1] for row in results]  # Extract the pulse_count (second element in each row)

min_count = min(pulse_counts)
max_count = max(pulse_counts)

# Logarithmic binning: Create bins from the minimum to the maximum value
log_bins = np.logspace(np.log10(min_count), np.log10(max_count), num=30)  # Adjust 'num' for bin count

# Plot the histogram with log bins
plt.figure(figsize=(10, 6))
plt.hist(pulse_counts, bins=log_bins, edgecolor='black')

# Set the x-axis to log scale
plt.xscale('log')
plt.yscale('log')

# Labels and title
plt.title('Histogram of Pulse Counts per Event (Log Binning)')
plt.xlabel('Number of Pulses per Event (log scale)')
plt.ylabel('Frequency')
plt.grid(True)

# Display the plot
plt.show()
plt.savefig('./dataset_stats/pulse_count_histogram.pdf')