# CIE2 – Activity 1: Multiple Linear Regression

## Project Overview
This project implements Multiple Linear Regression using Python to predict product sales based on advertising expenditure across TV, Radio, and Newspaper channels. It includes a manual normal equations derivation on a small sample, model training, comparison with a single-feature baseline, residual analysis, and evaluation on unseen test data.

## Objectives
- Understand Multiple Linear Regression.
- Perform data exploration and preprocessing.
- Solve normal equations manually on a sample and verify with library functions.
- Train and test single-feature and multiple-feature regression models.
- Calculate regression evaluation metrics (MAE, MSE, RMSE, R-squared, Adjusted R-squared).
- Check multicollinearity using VIF and analyze residuals.
- Predict sales for a new campaign budget.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy / Scikit-learn
- Google Colab / Jupyter Notebook

## Dataset
- **Dataset:** Advertising Dataset
- **Dataset Source:** https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv
- **Features:** TV, Radio, Newspaper (advertising budgets in thousands of dollars)
- **Target:** Sales (product sales in thousands of units)

## Work Completed
- Dataset loading and exploration.
- Data cleaning and preprocessing (dropped index column).
- Manual calculation worksheet using normal equations on 6 sample observations.
- Training and testing split (80:20).
- Single-feature regression baseline (Sales vs TV).
- Multiple-feature regression (Sales vs TV, Radio, Newspaper).
- Residual analysis and diagnostic visualizations.
- Multicollinearity check using Variance Inflation Factor (VIF).
- Model evaluation on unseen test data.
- Prediction on a new advertising budget.

## Model Evaluation Results

| Metric | Single-Feature Baseline (TV) | Multiple Linear Regression |
|---|---:|---:|
| MAE | 2.9200 | 1.2575 |
| MSE | 13.5526 | 2.5304 |
| RMSE | 3.6814 | 1.5907 |
| R-squared | 0.4947 | 0.9056 |
| Adjusted R-squared | 0.4814 | 0.8978 |

## Decision Log
I selected Multiple Linear Regression because product sales are influenced by multiple advertising media simultaneously. Adding Radio and Newspaper to TV spend reduced the test RMSE from 3.68 to 1.59 and increased R-squared from 0.49 to 0.91, showing that the multiple-feature model provides much better predictions than using TV alone.

## Submission Details
- **Submission Profile:** [https://github.com/complexiota](https://github.com/complexiota)

## Repository Contents
- `Activity_1_Multiple_Linear_Regression.ipynb` - Complete executable notebook with outputs
- `Advertising.csv` - Cleaned dataset
- `Manual_Calculation_Worksheet.md` - Step-by-step normal equations calculation on 6 sample records
- `README.md` - Project documentation
