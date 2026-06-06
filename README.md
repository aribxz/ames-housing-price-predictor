# Ames Housing Price Predictor

A machine learning project that predicts residential house prices using the Ames Housing dataset. Built as part of an ongoing journey learning machine learning — covers the full workflow from raw data to evaluated, tuned models.

---

## Project Overview

The goal was to build a regression model capable of predicting house sale prices given a set of property features. The Ames Housing dataset contains **2,930 houses** and **80 features** describing almost every aspect of a residential home — from basement quality to neighborhood to roof material.

This project covers:
- Data preprocessing and missing value treatment
- Encoding of ordinal and nominal categorical variables
- Training and evaluating multiple regression models
- Hyperparameter tuning with GridSearchCV and cross-validation
- Feature importance analysis using Lasso coefficients
- Comparison of models across multiple metrics

---

## Dataset

**Ames Housing Dataset** — compiled by Dean De Cock for data science education.

Download `train.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data), and place it inside the `data/` folder before running any scripts.

The dataset is not included in this repository.

---

## Project Structure

```
ames-housing-price-predictor/
│
├── data/
│   └── Model Algorithm Comparison Metrics.xlsx   # Results comparison across all models
│
├── Graphs/
│   ├── Actual Vs Predicted Plot.png              # Linear regression prediction quality
│   ├── Residual Plot.png                         # Residual distribution
│   ├── Ridge Alpha Tuning.png                    # GridSearchCV results for Ridge
│   ├── Lasso Alpha Tuning.png                    # GridSearchCV results for Lasso
│   └── Top 20 Features.jpeg                      # Feature importance via Lasso coefficients
│
├── files/
│   ├── Linear_Regression_Model.py                # Baseline model with plots and outlier analysis
│   ├── Ridge_Regression_Model.py                 # Final chosen model with tuned alpha
│   ├── Lasso_Regression_Model.py                 # Lasso with hyperparameter tuning and feature survival count
│   ├── Lasso_to_ridge.py                         # Lasso feature selection fed into Ridge
│   ├── Ridge_Vs_Lasso_(SS).py                    # Ridge and Lasso both with StandardScaler
│   └── Log_Ridge.py                              # Ridge with log-transformed target and StandardScaler
│
├── .gitignore
└── README.md
```

---

## Preprocessing Summary

The raw dataset required significant cleaning before any model could be trained:

- **Missing values** — handled differently based on context. Columns like `Pool QC` and `Fireplace Qu` had missing values that meant *no pool* or *no fireplace*, so they were filled with `"None"` or `0` rather than imputed. Genuinely missing values (e.g. one missing `Electrical` entry) were filled with the mode.
- **Ordinal encoding** — features with a natural ranking (e.g. `Kitchen Qual`: Poor → Fair → Average → Good → Excellent) were mapped to ordered integers to preserve the ranking relationship.
- **One-hot encoding** — nominal features with no inherent order (e.g. `Neighborhood`, `House Style`) were expanded into binary columns.
- **Result** — dataset grew from 82 columns to ~214 fully numerical columns with zero missing values.

---

## Models Trained

Six experiments were run to understand the effect of different approaches:

| Model | RMSE | MAE | R² (Test) | R² (Train) |
|---|---|---|---|---|
| Linear Regression | 36,494 | 18,474 | 0.83 | 0.92 |
| Ridge (α=10) | 30,641 | 18,168 | **0.8837** | 0.89 |
| Lasso | 30,866 | 18,284 | 0.88 | 0.89 |
| Lasso + StandardScaler | 32,178 | 18,644 | 0.87 | 0.88 |
| Ridge + StandardScaler | 30,600 | 18,189 | 0.88 | 0.90 |
| Ridge + Log Transform | 34,026 | 14,855 | 0.85 | 0.94 |
| Lasso → Ridge (L-R) | 30,838 | 18,287 | 0.88 | 0.89 |

**Final model: Ridge Regression with α=10**

---

## Key Findings

**1. Regularization significantly reduced overfitting.**
Linear Regression had a train/test R² gap of 0.92 vs 0.83 — a sign of overfitting on 214 features. Ridge closed this gap to 0.89 vs 0.8837, achieving better generalization.

**2. Neighborhood is the strongest price predictor.**
Lasso coefficient analysis revealed that `Neighborhood_StoneBr`, `Neighborhood_NridgHt`, and `Neighborhood_NoRidge` had by far the largest coefficients (~35,000–45,000), dwarfing structural features like Overall Quality and Kitchen Quality. Location matters more than the house itself.

**3. Lasso eliminated ~115 of 214 features with no meaningful loss.**
Lasso retained only ~99 features. When Ridge was retrained on only those 99 features (L-R experiment), R² dropped by just 0.0024 — confirming the eliminated features carried no real signal, and that Ridge's regularization was already suppressing them effectively.

**4. StandardScaler and log transformation produced negligible improvement.**
Both transformations were tested systematically. Neither produced meaningful gains over plain Ridge, which was an important practical finding about this dataset.

---

## Feature Importance

![Top 20 Features by Lasso Coefficient Magnitude](Graphs/Top%2020%20Features.jpeg)

The three most influential features are all neighborhood indicators, confirming that location dominates price prediction in this dataset. Roof material (`Roof Matl_WdShngl`) and second garage (`Misc Feature_Gar2`) also appeared unexpectedly high.

---

## How to Run

1. Clone the repository
2. Download `train.csv` from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place it in `data/`
3. Install dependencies:
```
pip install pandas numpy scikit-learn matplotlib openpyxl
```
4. Run any script from the `files/` folder:
```
python files/Ridge_Regression_Model.py
```

The preprocessing must be done on the data set first.

---

## Limitations

- R² of ~0.88 represents the ceiling for linear models on this dataset. Non-linear relationships (e.g. diminishing returns on square footage) cannot be captured by Ridge or Lasso.
- Moving to 0.92+ would require tree-based models such as Random Forest or gradient boosting (XGBoost), which are outside the scope of this project.
- The model was built as a learning exercise and is not intended for production use.

---

## What I Learned

- How to handle real-world messy data with 80 features and multiple types of missing values
- The difference between ordinal and nominal encoding and why it matters
- Why regularization (Ridge/Lasso) outperforms plain Linear Regression on high-dimensional data
- How GridSearchCV and cross-validation find optimal hyperparameters without overfitting to test data
- How Lasso's automatic feature selection reveals which features actually matter
- The importance of checking train vs test R² to diagnose overfitting — not just final accuracy
