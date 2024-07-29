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
event_nos = [event_no for event_no, in conn.execute(query, (min_count, max_count)).fetchall()]

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
