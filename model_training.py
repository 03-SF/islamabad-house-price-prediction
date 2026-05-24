"""
MACHINE LEARNING MODEL TRAINING & EVALUATION - IMPROVED VERSION
================================================================
Key improvements over original:
1. Random Forest tuned (200 trees, better depth/features)
2. XGBoost tuned with better hyperparameters
3. CatBoost tuned
4. All models benefit from improved features from preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False
import pickle
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 80)
print("MACHINE LEARNING MODEL TRAINING & EVALUATION")
print("=" * 80)

print("\n📂 Loading preprocessed data...")
X_train = pd.read_csv('X_train.csv')
X_test  = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test  = pd.read_csv('y_test.csv').values.ravel()

print(f"✅ Training set : {X_train.shape[0]} samples × {X_train.shape[1]} features")
print(f"✅ Testing set  : {X_test.shape[0]} samples × {X_test.shape[1]} features")

models  = {}
results = {}

def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    mae  = mean_absolute_error(y_te, preds)
    mse  = mean_squared_error(y_te, preds)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_te, preds)
    models[name]  = model
    results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R² Score': r2}
    print(f"✅ Training complete!")
    print(f"\n📊 Performance Metrics:")
    print(f"   MAE  : PKR {mae:,.0f}")
    print(f"   RMSE : PKR {rmse:,.0f}")
    print(f"   R²   : {r2:.4f}")

# ── Model 1: Linear Regression ────────────────────────────────────────────
print("\n" + "=" * 80)
print("MODEL 1: LINEAR REGRESSION")
print("=" * 80)
print("\n🔧 Training Linear Regression...")
evaluate('Linear Regression', LinearRegression(), X_train, y_train, X_test, y_test)

# ── Model 2: Decision Tree ────────────────────────────────────────────────
print("\n" + "=" * 80)
print("MODEL 2: DECISION TREE")
print("=" * 80)
print("\n🔧 Training Decision Tree...")
evaluate('Decision Tree',
         DecisionTreeRegressor(random_state=42, max_depth=12, min_samples_leaf=3),
         X_train, y_train, X_test, y_test)

# ── Model 3: Random Forest (tuned) ───────────────────────────────────────
print("\n" + "=" * 80)
print("MODEL 3: RANDOM FOREST  (tuned)")
print("=" * 80)
print("\n🔧 Training Random Forest (200 trees, tuned)...")
evaluate('Random Forest',
         RandomForestRegressor(
             n_estimators=200,
             max_depth=20,
             min_samples_leaf=2,
             max_features='sqrt',
             random_state=42,
             n_jobs=-1
         ),
         X_train, y_train, X_test, y_test)

# ── Model 4: Gradient Boosting ────────────────────────────────────────────
print("\n" + "=" * 80)
print("MODEL 4: GRADIENT BOOSTING")
print("=" * 80)
print("\n🔧 Training Gradient Boosting...")
evaluate('Gradient Boosting',
         GradientBoostingRegressor(
             n_estimators=200, learning_rate=0.08,
             max_depth=5, subsample=0.8, random_state=42
         ),
         X_train, y_train, X_test, y_test)

# ── Model 5: XGBoost (tuned) ──────────────────────────────────────────────
print("\n" + "=" * 80)
print("MODEL 5: XGBoost  (tuned)")
print("=" * 80)
print("\n🔧 Training XGBoost...")
evaluate('XGBoost',
         xgb.XGBRegressor(
             n_estimators=300, learning_rate=0.05,
             max_depth=6, subsample=0.8,
             colsample_bytree=0.8, min_child_weight=3,
             random_state=42, verbosity=0
         ),
         X_train, y_train, X_test, y_test)

# ── Model 6: CatBoost (tuned) ─────────────────────────────────────────────
print("\n" + "=" * 80)
print("MODEL 6: CatBoost  (tuned)")
print("=" * 80)
print("\n🔧 Training CatBoost...")
if CATBOOST_AVAILABLE:
    evaluate('CatBoost',
             cb.CatBoostRegressor(
                 iterations=300, learning_rate=0.05,
                 depth=6, l2_leaf_reg=3,
                 random_state=42, verbose=0
             ),
             X_train, y_train, X_test, y_test)
else:
    print("⚠️  CatBoost not installed — skipping")
    results['CatBoost'] = {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R² Score': np.nan}

# ── Comparison ────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("MODEL COMPARISON & RANKING")
print("=" * 80)

results_df = pd.DataFrame(results).T.round(2)
print("\n📊 PERFORMANCE COMPARISON TABLE:")
print(results_df.to_string())

best_name = results_df['R² Score'].idxmax()
best      = results_df.loc[best_name]

print("\n" + "=" * 80)
print("🏆 BEST PERFORMING MODEL")
print("=" * 80)
print(f"\n🥇 Winner: {best_name}")
print(f"\n📈 Metrics:")
print(f"   R² Score : {best['R² Score']:.4f}")
print(f"   MAE      : PKR {best['MAE']:,.0f}")
print(f"   RMSE     : PKR {best['RMSE']:,.0f}")

print(f"\n📊 RANKING BY R² SCORE:")
for i, (name, score) in enumerate(results_df['R² Score'].sort_values(ascending=False).items(), 1):
    print(f"   {i}. {name:25} → R² = {score:.4f}")

results_df.to_csv('model_results.csv')
print(f"\n💾 Results saved to model_results.csv")

with open('best_model.pkl', 'wb') as f:
    pickle.dump(models[best_name], f)
print(f"💾 Best model ({best_name}) saved as best_model.pkl")

with open('all_models.pkl', 'wb') as f:
    pickle.dump(models, f)
print(f"💾 All models saved as all_models.pkl")

print("\n" + "=" * 80)
print("✅ MODEL TRAINING & EVALUATION COMPLETED!")
print("=" * 80)
print(f"\n✨ Use {best_name} for predictions!")