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


# --- Feature Engineering: new derived columns ---
df["Fat_to_Carb_ratio"] = df["Fat (g)"] / (df["Carbohydrates (g)"] + 1)  # +1 avoids divide-by-zero
df["Protein_to_Fat_ratio"] = df["Protein (g)"] / (df["Fat (g)"] + 1)
df["Total_Macros"] = df["Protein (g)"] + df["Carbohydrates (g)"] + df["Fat (g)"]
df["Sugar_to_Carb_ratio"] = df["Sugars (g)"] / (df["Carbohydrates (g)"] + 1)

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
# --- Predicted vs Actual plot (using our best model: CatBoost) ---
best_preds = best_model.predict(X_test)

plt.figure(figsize=(7, 7))
plt.scatter(y_test, best_preds, alpha=0.4, color="teal")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)  # perfect-prediction line
plt.xlabel("Actual Calories")
plt.ylabel("Predicted Calories")
plt.title("CatBoost: Predicted vs Actual Calories")
plt.tight_layout()
plt.savefig("predicted_vs_actual.png")
print("Predicted vs Actual chart saved as predicted_vs_actual.png")
# --- Cross-validation on best model (CatBoost) ---
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
print("\n--- 5-Fold Cross-Validation (CatBoost) ---")
print("R2 scores per fold:", cv_scores)
print("Mean R2:", cv_scores.mean())
print("Std Dev:", cv_scores.std())# --- Hyperparameter tuning on CatBoost ---
from sklearn.model_selection import GridSearchCV

param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.03, 0.1, 0.2],
    'iterations': [200, 500]
}

tuned_model = CatBoostRegressor(verbose=0, random_state=42)

grid_search = GridSearchCV(
    estimator=tuned_model,
    param_grid=param_grid,
    scoring='r2',
    cv=3,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\n--- CatBoost Hyperparameter Tuning ---")
print("Best Params:", grid_search.best_params_)

best_tuned_model = grid_search.best_estimator_
tuned_preds = best_tuned_model.predict(X_test)
tuned_mae = mean_absolute_error(y_test, tuned_preds)
tuned_r2 = r2_score(y_test, tuned_preds)

print("Tuned MAE:", tuned_mae)
print("Tuned R2 Score:", tuned_r2)
print(f"\nComparison — Original CatBoost R2: {results[2][2]:.4f} vs Tuned CatBoost R2: {tuned_r2:.4f}")
# --- Stacking Ensemble ---
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

base_learners = [
    ("catboost", CatBoostRegressor(verbose=0, random_state=42, depth=4, iterations=500, learning_rate=0.1)),
    ("xgboost", XGBRegressor(random_state=42)),
    ("random_forest", RandomForestRegressor(random_state=42)),
]

# Meta-model: learns how to best combine the base models' predictions
meta_model = Ridge()

stacked_model = StackingRegressor(
    estimators=base_learners,
    final_estimator=meta_model,
    cv=5
)

stacked_model.fit(X_train, y_train)
stacked_preds = stacked_model.predict(X_test)

stacked_mae = mean_absolute_error(y_test, stacked_preds)
stacked_r2 = r2_score(y_test, stacked_preds)

print("\n--- Stacking Ensemble Results ---")
print("Stacked MAE:", stacked_mae)
print("Stacked R2 Score:", stacked_r2)
print(f"\nComparison — Tuned CatBoost R2: {tuned_r2:.4f} vs Stacked Ensemble R2: {stacked_r2:.4f}")