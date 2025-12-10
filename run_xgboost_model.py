"""
XGBoost Model Inference Script
Clean, reliable predictions using the XGBoost model
"""
import pandas as pd
import pickle
from arcgis.features import GeoAccessor

print("Loading XGBoost model package...")
with open('models/xgboost_accident_model.pkl', 'rb') as f:
    model_pkg = pickle.load(f)

model = model_pkg['model']
feature_names = model_pkg['feature_names']
cat_columns = model_pkg['categorical_columns']

print(f"✓ Loaded XGBoost model")
print(f"  Test Accuracy: {model_pkg['accuracy']:.2%}")
print(f"  Features: {feature_names}")

print("\nLoading test data...")
# Read only needed columns (avoids geometry dtype issue - XGBoost doesn't need geometry anyway)
columns_needed = feature_names + ['gravite_bin', 'id_accident']
test_df = pd.read_parquet('data/processed/paris_accidents_engineerd_4.parquet', columns=columns_needed).head(20000).copy()

# Prepare features (same order as training)
X = test_df[feature_names].copy()

# Convert categorical columns to 'category' dtype
for col in cat_columns:
    X[col] = X[col].astype('category')

print(f"\nRunning predictions on {len(X)} records...")
print("="*80)

# Predict
probabilities = model.predict_proba(X)

# Custom threshold for "Serious/Fatal"
THRESHOLD = 0.6433
predictions = (probabilities[:, 1] >= THRESHOLD).astype(int)

# Add results to dataframe
test_df['prediction'] = predictions
test_df['severity'] = test_df['prediction'].map({0: 'Minor', 1: 'Serious/Fatal'})
test_df['prob_minor'] = probabilities[:, 0]
test_df['prob_serious'] = probabilities[:, 1]

# Calculate accuracy
actuals = test_df['gravite_bin'].values
correct = (predictions == actuals).sum()
accuracy = correct / len(predictions) * 100

print("\n" + "="*80)
print("RESULTS:")
print("="*80)
print(f"Total records: {len(predictions)}")
print(f"Correct: {correct}")
print(f"Accuracy: {accuracy:.2f}%")

# Show distribution
pred_counts = pd.Series(predictions).value_counts().sort_index()
actual_counts = pd.Series(actuals).value_counts().sort_index()

print(f"\nPredicted distribution:")
print(f"  Minor (0): {pred_counts.get(0, 0)} ({pred_counts.get(0, 0)/len(predictions)*100:.1f}%)")
print(f"  Serious/Fatal (1): {pred_counts.get(1, 0)} ({pred_counts.get(1, 0)/len(predictions)*100:.1f}%)")

print(f"\nActual distribution:")
print(f"  Minor (0): {actual_counts.get(0, 0)} ({actual_counts.get(0, 0)/len(actuals)*100:.1f}%)")
print(f"  Serious/Fatal (1): {actual_counts.get(1, 0)} ({actual_counts.get(1, 0)/len(actuals)*100:.1f}%)")

# Show first 10 with probabilities
print(f"\nFirst 10 predictions:")
print(test_df[['id_accident', 'prediction', 'severity', 'gravite_bin', 
               'prob_minor', 'prob_serious']].head(200).to_string(index=False))

print("\n" + "="*80)
print("✓ Inference complete!")
print("="*80)
