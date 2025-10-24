# Code Documentation Guide

## Overview
This document explains each section of the `crop_yield_prediction_final.py` script in detail.

## 1. Data Loading and Preparation (Lines 1-54)

### Imports Section
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```
- **pandas**: Data manipulation and CSV reading
- **numpy**: Numerical computations
- **sklearn**: Machine learning algorithms and preprocessing
- **train_test_split**: Splits data into training and testing sets
- **StandardScaler**: Normalizes features to have mean=0, std=1
- **PolynomialFeatures**: Creates polynomial combinations of features
- **SVR**: Support Vector Regression
- **RandomForestRegressor**: Ensemble learning method
- **SelectKBest**: Feature selection based on statistical tests
- **Metrics**: Evaluation functions (RMSE, MAE, R²)

### Configuration
```python
DATA_FILE = 'lentil_data.csv' 
TARGET_COLUMN = 'Grain_Yield' 
RANDOM_SEED = 42
```
- Sets file paths and parameters for reproducibility

### Data Loading
```python
data = pd.read_csv(DATA_FILE)
X = data.drop(columns=[TARGET_COLUMN, 'Genotype_ID'])
y = data[TARGET_COLUMN]
```
- Loads CSV file
- Separates features (X) from target variable (y)
- Drops identifier column (Genotype_ID)

### Data Splitting
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)
```
- 80% training, 20% testing split
- Fixed random seed ensures reproducible results

### Data Scaling
```python
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
```
- Scales both features and target to prevent bias
- Fit on training data, transform on test data
- Reshape/flatten operations handle array dimensions

## 2. MARS-Like Feature Selection (Lines 55-88)

### Polynomial Feature Creation
```python
poly = PolynomialFeatures(degree=1, include_bias=False, interaction_only=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
```
- Creates polynomial combinations of features
- Degree=1 means linear terms only (like MARS max_degree=1)
- Captures non-linear relationships

### Statistical Feature Selection
```python
selector = SelectKBest(score_func=f_regression, k='all')
X_train_selected = selector.fit_transform(X_train_poly, y_train_scaled)
X_test_selected = selector.transform(X_test_poly)
```
- Uses F-regression to score features
- Selects all features (k='all') then filters by importance
- F-regression measures linear relationship strength

### Feature Importance Analysis
```python
importance_df = pd.DataFrame({
    'Feature': poly_feature_names, 
    'Score': feature_scores
}).sort_values(by='Score', ascending=False)
```
- Creates ranked list of feature importance
- Higher scores = more predictive features

## 3. Model Training and Evaluation (Lines 89-152)

### SVR Model
```python
svr_model = SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1)
svr_model.fit(X_train_selected, y_train_scaled)
```
- **kernel='rbf'**: Radial Basis Function kernel
- **C=10**: Regularization parameter (higher = less regularization)
- **gamma='scale'**: Kernel coefficient (auto-scaling)
- **epsilon=0.1**: Tolerance for prediction errors

### Random Forest Model
```python
rf_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
rf_model.fit(X_train_selected, y_train_scaled)
```
- **n_estimators=100**: Number of decision trees
- **random_state**: Ensures reproducible results

### Prediction and Evaluation
```python
y_pred_svr = scaler_y.inverse_transform(y_pred_svr_scaled.reshape(-1, 1)).flatten()
calculate_metrics(y_test, y_pred_svr, "MARS-Like SVR")
```
- Inverse transforms predictions back to original scale
- Calculates comprehensive evaluation metrics

## 4. Evaluation Metrics Explained

### RMSE (Root Mean Square Error)
- Measures average prediction error
- Penalizes large errors more heavily
- Lower is better

### MAE (Mean Absolute Error)
- Average absolute difference between predicted and actual
- Less sensitive to outliers than RMSE
- Lower is better

### MAPE (Mean Absolute Percentage Error)
- Percentage-based error metric
- Easy to interpret (10% MAPE = 10% average error)
- Lower is better

### R² (R-squared Score)
- Proportion of variance explained by the model
- Range: 0 to 1 (higher is better)
- 0.77 means model explains 77% of variance

## Key Design Decisions

1. **MARS-Like Approach**: Used polynomial features + statistical selection instead of true MARS due to installation constraints
2. **Feature Scaling**: Applied to both features and target for fair comparison
3. **Random Seed**: Fixed for reproducible results
4. **Train/Test Split**: 80/20 split follows machine learning best practices
5. **Model Selection**: SVR and Random Forest represent different learning paradigms
