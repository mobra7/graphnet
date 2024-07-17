import sqlite3

sqliteConnection = sqlite3.connect("/scratch/users/allorana/northern_sqlite/files_no_hlc/dev_northern_tracks_full_part_2.db")
Cursor = sqliteConnection.cursor()
Cursor.execute("""SELECT * FROM tum_dnn  ;""")
print([description[0] for description in Cursor.description])