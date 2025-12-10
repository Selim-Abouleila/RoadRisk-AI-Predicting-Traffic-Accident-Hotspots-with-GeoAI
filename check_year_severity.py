import pyarrow.parquet as pq
import pandas as pd

print("Reading 'annee' and 'gravite_bin' columns...")
# Read columns directly with PyArrow to avoid geometry issues
table = pq.read_table('data/processed/paris_accidents_engineerd_4.parquet', columns=['annee', 'gravite_bin'])
df = table.to_pandas()

# Calculate severity rate by year
severity_by_year = df.groupby('annee')['gravite_bin'].agg(['count', 'mean'])
severity_by_year['mean'] = severity_by_year['mean'] * 100  # Convert to percentage
severity_by_year.columns = ['Total Accidents', 'Severity Rate (%)']

print("\nAccident Severity by Year:")
print("="*50)
print(severity_by_year)
print("="*50)
