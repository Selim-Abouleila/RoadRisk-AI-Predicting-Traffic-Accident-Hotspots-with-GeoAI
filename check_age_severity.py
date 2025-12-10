import pandas as pd
import numpy as np

# Load data using GeoAccessor to handle geometry column correctly
from arcgis.features import GeoAccessor
df = pd.DataFrame.spatial.from_featureclass('data/processed/paris_accidents_engineerd_4.parquet')
df = df[['age', 'gravite_bin']].copy()

# Create age bins
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 25, 35, 50, 65, 100], 
                         labels=['0-18', '19-25', '26-35', '36-50', '51-65', '65+'])

# Calculate severity rate by age group
severity_by_age = df.groupby('age_group')['gravite_bin'].agg(['count', 'mean'])
severity_by_age['mean'] = severity_by_age['mean'] * 100  # Convert to percentage
severity_by_age.columns = ['Total Accidents', 'Severity Rate (%)']

print("\nAccident Severity by Age Group:")
print("="*50)
print(severity_by_age)
print("="*50)

# Check specific ages around 20
print("\nSeverity Rate for specific ages (18-25):")
subset = df[(df['age'] >= 18) & (df['age'] <= 25)]
age_stats = subset.groupby('age')['gravite_bin'].mean() * 100
print(age_stats)
