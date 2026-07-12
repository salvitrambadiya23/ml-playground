import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor  # New: adds "many trees combined" model
from catboost import CatBoostRegressor              # New: advanced boosting model
from xgboost import XGBRegressor                    # New: another advanced boosting model
from lightgbm import LGBMRegressor                   # New: fast boosting model, good for large data
# Load
df = pd.read_csv("data/daily_food_nutrition_dataset.csv", on_bad_lines='skip')

print("Shape:", df.shape)
print("\nColumns:", list(df.columns))

# Encode categorical columns
cat_cols = df.select_dtypes(include='object').columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Features and target
target = "Calories (kcal)"
X = df.drop(columns=[target])
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionary of models to compare — add/remove models here easily later
models = {
    "Decision Tree": DecisionTreeRegressor(random_state=42),   # your original model
    "Random Forest": RandomForestRegressor(random_state=42),   # NEW: ensemble of trees
    "CatBoost": CatBoostRegressor(verbose=0, random_state=42), # NEW: boosting, handles categories well
    "XGBoost": XGBRegressor(random_state=42),                  # NEW: fast, accurate boosting
    "LightGBM": LGBMRegressor(random_state=42),                # NEW: fast boosting, good for big data
}

results = []  # store (model_name, MAE, R2) for each model

# Loop through each model: train it, test it, print its score
for name, model in models.items():
    model.fit(X_train, y_train)              # train the model
    preds = model.predict(X_test)             # predict on unseen test data
    mae = mean_absolute_error(y_test, preds)  # average error in calories
    r2 = r2_score(y_test, preds)              # how well it explains variance
    results.append((name, mae, r2))
    print(f"\n--- {name} ---")
    print("MAE:", mae)
    print("R2 Score:", r2)

# Print a clean side-by-side comparison table at the end
print("\n\n=== Summary ===")
print(f"{'Model':<15}{'MAE':<12}{'R2 Score'}")
for name, mae, r2 in results:
    print(f"{name:<15}{mae:<12.4f}{r2:.4f}")
    # --- Feature Importance (using our best model: CatBoost) ---
import matplotlib.pyplot as plt

best_model = models["CatBoost"]  # reuse the already-trained CatBoost model
importances = best_model.get_feature_importance()
feature_names = X.columns

# Sort features by importance, highest first
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 6))
plt.barh([feature_names[i] for i in sorted_idx], [importances[i] for i in sorted_idx])
plt.xlabel("Importance")
plt.title("CatBoost Feature Importance — What Drives Calorie Predictions")
plt.gca().invert_yaxis()  # highest importance on top
plt.tight_layout()
plt.savefig("feature_importance.png")  # saves chart as an image file
print("\nFeature importance chart saved as feature_importance.png")