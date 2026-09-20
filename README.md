# Algerian Forest Fire — FWI Regression

Predicting the **Fire Weather Index (FWI)** for the Algerian Forest Fires dataset by training and comparing seven regression models: Linear Regression, Lasso, LassoCV, Ridge, RidgeCV, ElasticNet, and ElasticNetCV.

## Overview

`Model_Training.py` walks through a complete regression workflow:

1. **Load** `Algerian_forest_fires_cleaned_dataset.csv`.
2. **Clean up** — drop `day`, `month`, and `year` columns.
3. **Encode** — convert the categorical `Classes` column (`"fire"` / `"not fire"`) into binary `0`/`1`.
4. **Split features/target** — `X` is everything except `FWI`; `y` is `FWI`.
5. **Train/test split** — 75% train, 25% test (`random_state=42`).
6. **Multicollinearity check** — plot a correlation heatmap, then drop features with pairwise correlation above `0.85` (using a custom `correlation()` helper).
7. **Scale features** — `StandardScaler`, fit on training data only, applied to both train and test.
8. **Train & evaluate 7 models**, each reporting MAE, R², and a scatter plot of actual vs. predicted FWI:
   - Linear Regression
   - Lasso
   - LassoCV (5-fold, auto-selects best alpha)
   - Ridge
   - RidgeCV (5-fold)
   - ElasticNet
   - ElasticNetCV (5-fold, auto-selects best alpha/l1_ratio)

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install them with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Dataset

`Algerian_forest_fires_cleaned_dataset.csv` should be in the same directory as `Model_Training.py`. Expected columns include:

| Column | Description |
|---|---|
| day, month, year | Date fields (dropped before modeling) |
| Temperature | Temperature (°C) |
| RH | Relative humidity (%) |
| Ws | Wind speed (km/h) |
| Rain | Rainfall (mm) |
| FFMC | Fine Fuel Moisture Code |
| DMC | Duff Moisture Code |
| DC | Drought Code |
| ISI | Initial Spread Index |
| BUI | Buildup Index |
| FWI | Fire Weather Index — **target** |
| Classes | `fire` / `not fire` — encoded to 1/0 |
| Region | Region indicator |

## Usage

```bash
python Model_Training.py
```

Running it will print dataset previews, the correlation matrix, dropped high-correlation features, and per-model MAE/R² scores, while displaying: a correlation heatmap, before/after scaling boxplots, and an actual-vs-predicted scatter plot for each of the 7 models.

## Key Design Notes

- **Scaling**: `fit_transform` is applied only to `X_train`; `X_test` uses `transform` with statistics learned from training, avoiding data leakage.
- **Multicollinearity handling**: the `correlation()` helper flags any feature pair correlated above `0.85` and drops one from each pair — this is done before scaling and modeling to keep the feature set more interpretable and stable, especially for the non-regularized Linear Regression model.
- **Why compare Lasso/Ridge/ElasticNet**: these add L1, L2, and combined L1+L2 penalties respectively to control overfitting and (for Lasso/ElasticNet) perform implicit feature selection by shrinking some coefficients to zero.
- **CV variants (LassoCV, RidgeCV, ElasticNetCV)**: automatically search over a range of alpha values via cross-validation to pick the regularization strength that generalizes best, rather than relying on the default `alpha=1.0`.

## Possible Next Steps

- Tabulate MAE/R² across all 7 models side by side for a direct comparison instead of reading them from separate print statements.
- Inspect model coefficients (especially Lasso/ElasticNet) to see which features got zeroed out.
- Try `GridSearchCV`/`RandomizedSearchCV` for more exhaustive hyperparameter tuning beyond the CV variants' built-in alpha search.
- Save the trained model and scaler with `joblib` for reuse without retraining.
- Add a classification model for the `Classes` (`fire`/`not fire`) target as a complementary task alongside the FWI regression.