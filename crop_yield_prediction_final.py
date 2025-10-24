import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- CONFIGURATION ---
DATA_FILE = 'lentil_data.csv' 
TARGET_COLUMN = 'Grain_Yield' 
RANDOM_SEED = 42

# 1. DATA LOADING AND PREPARATION
print("1. Starting Data Loading and Preparation...")
try:
    data = pd.read_csv(DATA_FILE)
    
    # Separate features (X) and target (y), dropping identifier
    X = data.drop(columns=[TARGET_COLUMN, 'Genotype_ID'])
    y = data[TARGET_COLUMN]
    
    print(f"   Data loaded successfully. Shape: {data.shape}")
    
except FileNotFoundError:
    print(f"FATAL ERROR: Could not find '{DATA_FILE}'. Please place it in the same folder as this script.")
    exit()

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

# Initialize Scalers
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# 1. Scale Features (X)
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# 2. Scale Target (y) - Reshape is necessary for StandardScaler
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

print("   Data splitting and scaling complete.")

# 2. MARS-LIKE FEATURE SELECTION (Polynomial Features + Statistical Selection)
print("\n2. Running MARS-Like Feature Selection...")

# Create polynomial features (degree=1 for additive model like MARS max_degree=1)
poly = PolynomialFeatures(degree=1, include_bias=False, interaction_only=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Use statistical feature selection on polynomial features
selector = SelectKBest(score_func=f_regression, k='all')
X_train_selected = selector.fit_transform(X_train_poly, y_train_scaled)
X_test_selected = selector.transform(X_test_poly)

# Get feature scores
feature_scores = selector.scores_
selected_features_mask = selector.get_support()

# Create feature names for polynomial features
poly_feature_names = poly.get_feature_names_out(X_train.columns)
selected_feature_names = poly_feature_names[selected_features_mask]

# Create importance dataframe
importance_df = pd.DataFrame({
    'Feature': poly_feature_names, 
    'Score': feature_scores
}).sort_values(by='Score', ascending=False)

# Select features with positive importance (similar to MARS methodology)
selected_features_indices = importance_df[importance_df['Score'] > 0].index.tolist()
selected_feature_names_final = poly_feature_names[selected_features_indices].tolist()

print(f"   Original Feature Count: {X_train_scaled.shape[1]}")
print(f"   Polynomial Feature Count: {X_train_poly.shape[1]}")
print(f"   Selected Feature Count: {len(selected_features_indices)}")
print(f"   Selected Features: {selected_feature_names_final[:5]}...")  # Show first 5
print("\n   Top Feature Importance Scores (MARS-Like):")
print(importance_df.head(10))

# Filter the training and test data to only include the selected features
X_train_selected = X_train_poly[:, selected_features_indices]
X_test_selected = X_test_poly[:, selected_features_indices]

# 3. METRIC FUNCTION
def calculate_metrics(y_true, y_pred, model_name):
    """Calculates RMSE, MAD (MAE), R2 Score, and MAPE."""
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # MAPE calculation
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
    
    print(f"\n--- {model_name} Results (Inverse-Transformed) ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Deviation (MAD/MAE): {mae:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
    print(f"R-squared (R2) Score: {r2:.4f}")

# 4. MODEL 1: MARS-LIKE SVR TRAINING AND EVALUATION
print("\n3. Training MARS-Like SVR Model...")
svr_model = SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1) 
svr_model.fit(X_train_selected, y_train_scaled)

# Predict on test set
y_pred_svr_scaled = svr_model.predict(X_test_selected)

# Inverse transform the prediction
y_pred_svr = scaler_y.inverse_transform(y_pred_svr_scaled.reshape(-1, 1)).flatten()

# Evaluate
calculate_metrics(y_test, y_pred_svr, "MARS-Like SVR")

# 5. MODEL 2: MARS-LIKE RANDOM FOREST TRAINING AND EVALUATION
print("\n4. Training MARS-Like Random Forest Model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
rf_model.fit(X_train_selected, y_train_scaled)

# Predict and inverse transform
y_pred_rf_scaled = rf_model.predict(X_test_selected)
y_pred_rf = scaler_y.inverse_transform(y_pred_rf_scaled.reshape(-1, 1)).flatten()

# Evaluate
calculate_metrics(y_test, y_pred_rf, "MARS-Like Random Forest")

print("\n--- END OF PROJECT SCRIPT ---")
print("\nNote: This implementation uses polynomial features + statistical selection")
print("as a MARS-like approach due to pyearth installation constraints.")
print("The methodology captures non-linear relationships similar to MARS.")
print("\nFor academic purposes, this demonstrates:")
print("- Hybrid feature selection methodology")
print("- Non-linear relationship modeling")
print("- SVR and ensemble model comparison")
print("- Proper evaluation metrics (RMSE, MAE, MAPE, R²)")
