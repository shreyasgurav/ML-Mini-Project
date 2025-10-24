# Crop Yield Prediction ML Project

## Installation Instructions

1. Create and activate virtual environment:
```bash
python3 -m venv ml_env
source ml_env/bin/activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the main script:
```bash
python crop_yield_prediction_final.py
```

## Project Overview

This project implements a **MARS-Like Hybrid Machine Learning Approach** for lentil crop yield prediction, inspired by the research paper "Crop Yield Prediction Using Hybrid Machine Learning Approach."

### Dataset:
- **550 lentil samples** with 9 phenotypic features
- **Target**: Grain Yield (kg/ha)
- **Train/Test Split**: 80%/20% with fixed random seed (42)

### Features Used:
- **DF** (Days to Flowering)
- **PH** (Plant Height) 
- **DM** (Days to Maturity)
- **SW** (Seed Weight)
- **BYP** (Biological Yield per Plant) - *Most Important*
- **PB** (Pods per Branch)
- **SB** (Seeds per Branch)
- **PPP** (Pods per Plant)
- **HIN** (Harvest Index)

### Methodology:
1. **MARS-Like Feature Selection**: Polynomial features + Statistical selection
2. **Data Scaling**: StandardScaler for features and target
3. **Hybrid Models**: Feature-selected SVR and Random Forest

### Models Implemented:
1. **MARS-Like SVR**: Support Vector Regression with RBF kernel
2. **MARS-Like Random Forest**: Ensemble method with 100 trees

### Results:
- **MARS-Like SVR**: RMSE=2.07, R²=0.69, MAPE=10.7%
- **MARS-Like Random Forest**: RMSE=1.79, R²=0.77, MAPE=9.3%

### Key Findings:
- **BYP** (Biological Yield per Plant) is the most predictive feature
- **Random Forest outperforms SVR** with 77% accuracy
- **Low MAPE** (~9-10%) indicates good prediction accuracy

### Output Metrics:
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error) 
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (R-squared Score)

### Academic Value:
- Demonstrates hybrid machine learning methodology
- Captures non-linear relationships in agricultural data
- Provides reproducible results for crop yield prediction
- Suitable for lab project submission
