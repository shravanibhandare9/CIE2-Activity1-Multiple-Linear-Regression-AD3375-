# CIE 2 Activity 1: Multiple Linear Regression Worksheet
**Department:** Artificial Intelligence and Data Science | **Class:** T.Y. B.Tech  
**Course:** Regression and Classification Algorithms | **Semester:** V (Academic Year: 2026–27)  
**Institution:** Zeal Education Society's Zeal College of Engineering and Research, Narhe, Pune  
**Record No.:** ZCOER-ACAD/R/16K | **Marks:** 5  

---

- **Faculty Submission Evaluation Profile:** [https://github.com/complexiota](https://github.com/complexiota)

---

## 1. Project Overview & Professional Framing
As a marketing data analyst within an organization, the objective is to model and forecast product `Sales` from advertising expenditure across three media channels (`TV`, `Radio`, and `Newspaper`). Rather than focusing solely on fitting a formula, a professional data scientist frames the problem around resource allocation, tests for multicollinearity, guards against data leakage, compares against a baseline benchmark, verifies manual mechanics via normal equations, and evaluates predictive accuracy on unseen test data.

---

## 2. Dataset Description and Source
- **Dataset:** Advertising Dataset (200 market records)
- **Source Citation:** *An Introduction to Statistical Learning* (James, Witten, Hastie, Tibshirani)  
  Direct CSV link: [Advertising.csv](https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv)
- **Input Features ($X$):**
  - `TV`: Television advertising budget (thousands of USD)
  - `Radio`: Radio advertising budget (thousands of USD)
  - `Newspaper`: Print newspaper advertising budget (thousands of USD)
- **Target Variable ($y$):**
  - `Sales`: Product sales volume (thousands of units)
- **Data Cleaning:** The non-predictive 1-based index column `Unnamed: 0` was identified and removed. Zero missing values and zero duplicate rows were verified.

---

## 3. Decision Log
| Decision | Alternatives Considered | Professional Rationale |
|:---|:---|:---|
| **Drop `Unnamed: 0` column** | Retain as feature or index | Identifier numbers contain no real-world market signal and risk spurious correlations. |
| **Retain all 3 channels in MLR** | Drop `Newspaper` upfront | Kept to evaluate incremental Adjusted $R^2$; VIF confirmed absence of harmful multicollinearity. |
| **Leakage-free Standardization** | Scale entire dataset before split | Fit scaler strictly on 160 training rows; prevented test-set distribution statistics from leaking into training. |
| **Model Selection Benchmark** | Mean baseline & Single-feature | Required multiple model to prove that adding features meaningfully beat the single-feature baseline. |

---

## 4. Manual Calculation Worksheet Summary (Sample $n=6$)
To demonstrate the mathematical mechanics of Ordinary Least Squares (OLS) regression, a 6-observation sample was solved by hand using normal equations:

$$\mathbf{b} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

- **Design Matrix $\mathbf{X}$ ($6 \times 3$):** Prepending a column of ones for intercept with `TV` ($X_1$) and `Radio` ($X_2$).
- **Intermediate Matrices:**
  $$\mathbf{X}^T \mathbf{X} = \begin{bmatrix} 6.0 & 632.8 & 224.0 \\ 632.8 & 110938.68 & 19871.13 \\ 224.0 & 19871.13 & 9293.68 \end{bmatrix}, \quad \mathbf{X}^T \mathbf{y} = \begin{bmatrix} 80.4 \\ 10905.68 \\ 2926.42 \end{bmatrix}$$
- **Solved Manual Coefficients:**
  $$b_0 = -2.281725, \quad b_1 (\text{TV}) = 0.073039, \quad b_2 (\text{Radio}) = 0.213710$$
- **Fitted Sample Equation:**
  $$\widehat{y} = -2.281725 + 0.073039 \cdot \text{TV} + 0.213710 \cdot \text{Radio}$$
- **Sample Prediction (Observation 1: TV=230.1, Radio=37.8, Actual Sales=22.1):**
  $$\widehat{y}_1 = 22.602884 \implies \text{Residual } e_1 = 22.1 - 22.602884 = \mathbf{-0.502884} \text{ thousand units (-503 units)}$$
- **Library Verification:** `numpy.linalg.lstsq` reproduced the exact coefficients with a maximum absolute discrepancy of $8.26 \times 10^{-14}$.

*(See [Manual_Calculation_Worksheet.md](Manual_Calculation_Worksheet.md) for complete step-by-step arithmetic.)*

---

## 5. Full Multiple Regression Model & Coefficient Interpretation
On the 80% training split (160 observations):

$$\mathbf{\text{Sales} = 3.0389 + 0.046363 \cdot \text{TV} + 0.175484 \cdot \text{Radio} + 0.001084 \cdot \text{Newspaper}}$$

- **Intercept ($b_0 = 3.0389$):** Expected baseline sales of ~3,039 units when all advertising budgets are zero. Acts as a mathematical anchor.
- **TV ($b_1 = +0.046363$):** Each additional \$1,000 spent on TV increases sales by **46.4 units**, holding Radio and Newspaper constant.
- **Radio ($b_2 = +0.175484$):** Each additional \$1,000 spent on Radio increases sales by **175.5 units**, holding TV and Newspaper constant.
- **Newspaper ($b_3 = +0.001084$):** Adds only **1.1 units** per \$1,000 spent, showing virtually no explanatory contribution.
- **Standardized Coefficients (Relative Influence):**
  - TV: $\beta^* = \mathbf{+3.9377}$ (Dominant driver)
  - Radio: $\beta^* = \mathbf{+2.6197}$ (Strong secondary driver)
  - Newspaper: $\beta^* = \mathbf{+0.0231}$ (Negligible impact)

---

## 6. Model Evaluation: Single-Feature vs. Multiple Regression
Evaluation on the 20% unseen test set (40 observations):

| Evaluation Metric | Single-Feature Baseline (`TV`) | Multiple Regression (`TV`, `Radio`, `Newspaper`) | Performance Delta / Justification |
|:---|:---:|:---:|:---|
| **MAE** | 2.9200 | **1.2575** | Reduced average absolute error by **56.9%** |
| **MSE** | 13.5526 | **2.5304** | Variance of errors dropped by **81.3%** |
| **RMSE** | 3.6814 | **1.5907** | Out-of-sample error slashed from 3,681 to **1,591 units** |
| **Test $R^2$** | 0.4947 | **0.9056** | Explains **90.56%** of sales variance (vs. 49.47%) |
| **Adjusted $R^2$** | 0.4814 | **0.8978** | Penalized complexity confirms extra features are strongly justified |

---

## 7. Multicollinearity & Diagnostic Analysis
- **Variance Inflation Factors (VIF):**
  - TV: **1.0079** ($R^2_{\text{others}} = 0.0079$)
  - Radio: **1.1979** ($R^2_{\text{others}} = 0.1652$)
  - Newspaper: **1.2066** ($R^2_{\text{others}} = 0.1712$)  
  *Finding:* All VIF values are well below 5, confirming no severe multicollinearity among predictors.
- **Residual Diagnostics:**
  - Residuals vs. Predicted values are evenly distributed around zero.
  - Residual histogram is approximately bell-shaped and symmetric around zero error.
  - Predicted vs. Actual values closely follow the ideal 45-degree line ($y = x$).

---

## 8. Prediction for a New Campaign Example
For a planned quarterly campaign within the historical range:
- Budget: `TV = $150.0k`, `Radio = $25.0k`, `Newspaper = $20.0k`
- **Predicted Sales:** **14.4020 thousand units** (**14,402 units**)

---

## 9. Limitations & Ethical Considerations
1. **Synergy / Non-Linearity:** The additive linear model assumes channels do not interact. In reality, TV and Radio campaigns often show cross-channel reinforcement.
2. **Diminishing Marginal Returns:** At extreme spending levels, advertising effectiveness saturates.
3. **Extrapolation Limits:** Predictions must not be made outside the observed budget boundaries (e.g., $\text{TV} > \$300\text{k}$).
4. **Causation vs Correlation:** Historical data captures observational correlation; controlled experimentation is necessary before claiming causal attribution.

---

## 10. Repository Structure
```text
├── Activity_1_Multiple_Linear_Regression.ipynb   # Executable Jupyter Notebook with all outputs
├── Advertising.csv                               # Cleaned dataset (200 records)
├── Manual_Calculation_Worksheet.md               # Step-by-step matrix derivation & verification
├── Activity_1_Comprehensive_Report.md            # Detailed 34-section technical report
├── requirements.txt                              # Environment dependencies
├── .gitignore                                    # Git ignore rules
└── README.md                                     # Main project documentation (this file)
```

---

## 11. Instructions to Run
```bash
# Clone the repository
git clone https://github.com/[your-username]/[your-repo-name].git
cd [your-repo-name]

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Lab / Notebook
jupyter lab
```
Or open `Activity_1_Multiple_Linear_Regression.ipynb` directly in [Google Colab](https://colab.research.google.com).


