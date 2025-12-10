import pyarrow.parquet as pq
import numpy as np

print("Reading 'age' column from CLEANED (unscaled) data...")
# Read from cleaned parquet which should have raw age values
table = pq.read_table('data/processed/paris_accidents_cleaned.parquet', columns=['age'])

# Convert just that column to numpy (bypassing pandas metadata check for the whole table)
age_array = table['age'].to_numpy()

print(f"\nLoaded {len(age_array)} records.")

# Calculate stats
mean_val = np.nanmean(age_array)
std_val = np.nanstd(age_array)

print(f"\nCalculated Stats:")
print(f"MEAN_AGE = {mean_val:.4f}")
print(f"STD_AGE = {std_val:.4f}")
