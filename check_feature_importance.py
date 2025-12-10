import pickle
import pandas as pd

print("Loading XGBoost model...")
with open('models/xgboost_accident_model.pkl', 'rb') as f:
    model_pkg = pickle.load(f)

model = model_pkg['model']
feature_names = model_pkg['feature_names']

# Get feature importance
# XGBoost provides 'gain', 'weight', 'cover', 'total_gain', 'total_cover'
# Default feature_importances_ is usually 'gain' (contribution to accuracy)
importance = model.feature_importances_
feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
feat_imp = feat_imp.sort_values('importance', ascending=False)

print("\nFeature Importance (Contribution to Prediction):")
print("="*60)
print(feat_imp.to_string(index=False))
print("="*60)
