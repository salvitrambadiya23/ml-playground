# ML Playground

A repository for machine learning projects, models, and experimentation.

## Directory Structure

- `TREE MODELS/` - Models and notebooks relating to tree-based algorithms (Decision Trees, Random Forests, XGBoost, etc.).

# Food Calorie Prediction

A machine learning project that predicts food calorie content from nutritional data (protein, carbs, fat, etc.) by training and comparing multiple regression models to find the most accurate predictor.

## Dataset
Daily Food & Nutrition Dataset (Kaggle) — includes calories, protein, carbohydrates, fat, fiber, sugar, sodium, cholesterol, meal type, and water intake per food entry.

## Approach
1. **Data cleaning** — handled malformed CSV rows during parsing
2. **Feature engineering** — encoded categorical columns, added derived features (Fat_to_Carb_ratio, Protein_to_Fat_ratio, Total_Macros, Sugar_to_Carb_ratio)
3. **Model comparison** — trained and evaluated 5 models: Decision Tree, Random Forest, CatBoost, XGBoost, LightGBM
4. **Hyperparameter tuning** — GridSearchCV on CatBoost (best model)
5. **Cross-validation** — 5-fold CV to confirm result stability
6. **Stacking ensemble** — combined CatBoost, XGBoost, and Random Forest with a Ridge meta-model for the final best result

## Results

| Model | R² Score |
|---|---|
| Decision Tree (baseline) | 0.9368 |
| Random Forest | 0.9841 |
| CatBoost (default) | 0.9856 |
| XGBoost | 0.9805 |
| LightGBM | 0.9786 |
| CatBoost + feature engineering | 0.9881 |
| CatBoost (tuned) | 0.9885 |
| **Stacking Ensemble (best)** | **0.9891** |

Validated with 5-fold cross-validation: mean R² = 0.987, std dev = 0.005 — confirming stable, reliable performance rather than a lucky split.

## Key Insight
Feature importance analysis showed **Fat** and **Carbohydrates** are the dominant predictors of calorie content — consistent with nutritional science, since fat contributes ~9 calories/gram versus ~4 for protein and carbs. Specific food identity and meal timing contributed almost nothing once macronutrients were known, indicating the model learned genuine nutritional relationships rather than memorizing food labels.

## Visualizations
- `feature_importance.png` — which nutrients matter most
- `predicted_vs_actual.png` — model prediction accuracy visualization

## How to Run
```
pip install -r requirements.txt
py train.py
```