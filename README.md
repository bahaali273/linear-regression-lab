# Linear Regression and Model Evaluation

A reproducible Jupyter Notebook for a Data Science lab using the **scikit-learn Diabetes regression dataset** (442 observations, 10 numerical predictors, and a quantitative disease-progression target).

## Project overview

This lab explores linear relationships with the target, compares a simple BMI-only linear regression model with a multiple linear regression model using all 10 predictors, evaluates both against a mean-prediction baseline, checks residual patterns, and performs 5-fold cross-validation on the training set.

## Contents

- `Linear_Regression_Lab.ipynb` — complete notebook with code, labeled figures, analyses, written responses, and a 166-word conclusion.
- `requirements.txt` — Python package dependencies.
- `README.md` — setup instructions and project summary.

The dataset loads directly using `sklearn.datasets.load_diabetes(as_frame=True)`; no external dataset download is required.

## Setup and execution

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Linear_Regression_Lab.ipynb
```

In Jupyter, use **Kernel → Restart Kernel and Run All Cells**. Alternatively, upload the `.ipynb` to Google Colab and use **Runtime → Restart session and run all**. Run cells in their original order.

## Methodology

1. Inspect feature and target distributions, missing values, descriptive statistics, BMI scatter plot, and correlation heatmap.
2. Split the data into **80% training / 20% testing** using `random_state=42`.
3. Fit a simple linear regression using scaled BMI as the only predictor.
4. Compare its MAE, RMSE, and R² with a `DummyRegressor(strategy="mean")` baseline.
5. Fit multiple linear regression with all 10 numerical predictors using the same train/test split.
6. Inspect actual vs predicted values, residuals vs predictions, and a residual histogram.
7. Run **5-fold cross-validation on the training subset only**, using R² as the score.

## Results

| Model | Test MAE ↓ | Test RMSE ↓ | Test R² ↑ |
|---|---:|---:|---:|
| Mean baseline | 64.0065 | 73.2225 | -0.0120 |
| Simple linear regression (BMI) | 52.2600 | 63.7325 | 0.2334 |
| Multiple linear regression (10 features) | **42.7941** | **53.8534** | **0.4526** |

The three strongest absolute correlations with the target were **BMI (0.5865)**, **s5 (0.5659)**, and **bp (0.4415)**.

Five-fold cross-validation (multiple regression, training set only):

- Fold R² scores: **0.5476, 0.3612, 0.5048, 0.6177, 0.2149**
- Mean R²: **0.4493**
- Standard deviation: **0.1441**

These results favor multiple linear regression for this dataset and split, though predictive power is moderate and varies across validation folds.

## Limitations

- Linear models may miss nonlinear relationships and interactions.
- Some features are correlated; individual coefficients should not be interpreted causally.
- The dataset is relatively small, and cross-validation scores vary between folds.
- Results are descriptive and predictive, not clinical guidance.

## Reproducibility notes

- No API keys, external data files, or paid services are needed.
- Data splitting uses `random_state=42` for reproducibility.
- Small metric differences can occur across different library versions.
- The test set remains separate from cross-validation.

## Submission

The assignment requires **one reproducible `.ipynb` notebook** with plots, written answers, a model-comparison table, and a **150–200-word conclusion**. The README and requirements file are additional GitHub documentation, not substitutes for the notebook.
