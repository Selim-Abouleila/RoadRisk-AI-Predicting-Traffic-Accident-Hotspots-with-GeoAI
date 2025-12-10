import pandas as pd
import numpy as np

# Load data - read only 'age' to avoid geometry issues
# We must ensure we don't trigger the geometry metadata read
try:
    df = pd.read_parquet('data/processed/paris_accidents_engineerd_4.parquet', columns=['age'])
except Exception as e:
    # Fallback: try reading with pyarrow directly if pandas fails
    import pyarrow.parquet as pq
    table = pq.read_table('data/processed/paris_accidents_engineerd_4.parquet', columns=['age'])
    df = table.to_pandas()

print("Age Statistics:")
print(df['age'].describe())

print("\nSample values:")
print(df['age'].head(10).values)
