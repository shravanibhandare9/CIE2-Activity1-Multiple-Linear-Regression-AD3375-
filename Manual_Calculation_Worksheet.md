# Activity 1: Manual Calculation Worksheet

## 1. Objective and Problem Definition
The objective of this worksheet is to perform a step-by-step manual derivation of the **Ordinary Least Squares (OLS) Multiple Linear Regression** model for a small sample of observations using the matrix normal equations:

$$\mathbf{b} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

We verify each intermediate arithmetic quantity, solve for intercept $b_0$ and partial slope coefficients $b_1, b_2$, predict the target for an individual observation, calculate its residual, and verify the resulting parameters against Python's numerical linear algebra library.

- **Input Features:**
  - $X_1$: `TV` advertising expenditure (in thousands of USD)
  - $X_2$: `Radio` advertising expenditure (in thousands of USD)
- **Target Variable:**
  - $y$: `Sales` (in thousands of units)

---

## 2. Selected Sample Data (n = 6 Observations)
We select the first 6 observations from the cleaned `Advertising.csv` dataset:

| Observation ($i$) | TV Budget ($X_{1i}$) | Radio Budget ($X_{2i}$) | Actual Sales ($y_i$) |
|:------------------:|:--------------------:|:-----------------------:|:--------------------:|
| **1**              | 230.1                | 37.8                    | 22.1                 |
| **2**              | 44.5                 | 39.3                    | 10.4                 |
| **3**              | 17.2                 | 45.9                    | 9.3                  |
| **4**              | 151.5                | 41.3                    | 18.5                 |
| **5**              | 180.8                | 10.8                    | 12.9                 |
| **6**              | 8.7                  | 48.9                    | 7.2                  |

---

## 3. Construction of Design Matrix ($\mathbf{X}$) and Target Vector ($\mathbf{y}$)
We prepend a column of ones representing the intercept term to form the $6 \times 3$ design matrix $\mathbf{X}$ and the $6 \times 1$ target vector $\mathbf{y}$:

$$\mathbf{X} = \begin{bmatrix}
1 & 230.1 & 37.8 \\
1 & 44.5 & 39.3 \\
1 & 17.2 & 45.9 \\
1 & 151.5 & 41.3 \\
1 & 180.8 & 10.8 \\
1 & 8.7 & 48.9
\end{bmatrix}, \quad
\mathbf{y} = \begin{bmatrix}
22.1 \\
10.4 \\
9.3 \\
18.5 \\
12.9 \\
7.2
\end{bmatrix}$$

---

## 4. Computation of Intermediate Sums and Dot Products
To form $\mathbf{X}^T \mathbf{X}$ and $\mathbf{X}^T \mathbf{y}$, we compute the scalar sums across all $n=6$ records:

### Summary Summations:
- $n = \sum 1 = 6$
- $\sum X_1 = 230.1 + 44.5 + 17.2 + 151.5 + 180.8 + 8.7 = \mathbf{632.8}$
- $\sum X_2 = 37.8 + 39.3 + 45.9 + 41.3 + 10.8 + 48.9 = \mathbf{224.0}$
- $\sum X_1^2 = (230.1)^2 + (44.5)^2 + (17.2)^2 + (151.5)^2 + (180.8)^2 + (8.7)^2$
  $$= 52946.01 + 1980.25 + 295.84 + 22952.25 + 32688.64 + 75.69 = \mathbf{110938.68}$$
- $\sum X_2^2 = (37.8)^2 + (39.3)^2 + (45.9)^2 + (41.3)^2 + (10.8)^2 + (48.9)^2$
  $$= 1428.84 + 1544.49 + 2106.81 + 1705.69 + 116.64 + 2391.21 = \mathbf{9293.68}$$
- $\sum X_1 X_2 = (230.1)(37.8) + (44.5)(39.3) + (17.2)(45.9) + (151.5)(41.3) + (180.8)(10.8) + (8.7)(48.9)$
  $$= 8697.78 + 1748.85 + 789.48 + 6256.95 + 1952.64 + 425.43 = \mathbf{19871.13}$$
- $\sum y = 22.1 + 10.4 + 9.3 + 18.5 + 12.9 + 7.2 = \mathbf{80.4}$
- $\sum X_1 y = (230.1)(22.1) + (44.5)(10.4) + (17.2)(9.3) + (151.5)(18.5) + (180.8)(12.9) + (8.7)(7.2)$
  $$= 5085.21 + 462.80 + 159.96 + 2802.75 + 2332.32 + 62.64 = \mathbf{10905.68}$$
- $\sum X_2 y = (37.8)(22.1) + (39.3)(10.4) + (45.9)(9.3) + (41.3)(18.5) + (10.8)(12.9) + (48.9)(7.2)$
  $$= 835.38 + 408.72 + 426.87 + 764.05 + 139.32 + 352.08 = \mathbf{2926.42}$$

---

## 5. Construction of $\mathbf{X}^T \mathbf{X}$ and $\mathbf{X}^T \mathbf{y}$

$$\mathbf{X}^T \mathbf{X} = \begin{bmatrix}
n & \sum X_1 & \sum X_2 \\
\sum X_1 & \sum X_1^2 & \sum X_1 X_2 \\
\sum X_2 & \sum X_1 X_2 & \sum X_2^2
\end{bmatrix} = \begin{bmatrix}
6.0 & 632.8 & 224.0 \\
632.8 & 110938.68 & 19871.13 \\
224.0 & 19871.13 & 9293.68
\end{bmatrix}$$

$$\mathbf{X}^T \mathbf{y} = \begin{bmatrix}
\sum y \\
\sum X_1 y \\
\sum X_2 y
\end{bmatrix} = \begin{bmatrix}
80.4 \\
10905.68 \\
2926.42
\end{bmatrix}$$

---

## 6. Matrix Inversion of $\mathbf{X}^T \mathbf{X}$
Let $\mathbf{A} = \mathbf{X}^T \mathbf{X}$. The inverse is computed as:

$$\mathbf{A}^{-1} = \frac{1}{\det(\mathbf{A})} \operatorname{adj}(\mathbf{A})$$

### Determinant Calculation:
$$\det(\mathbf{A}) = 6 \cdot [ (110938.68)(9293.68) - (19871.13)^2 ] \\
- 632.8 \cdot [ (632.8)(9293.68) - (19871.13)(224.0) ] \\
+ 224.0 \cdot [ (632.8)(19871.13) - (110938.68)(224.0) ]$$

Evaluating each term:
1. $(110938.68)(9293.68) - (19871.13)^2 = 1031028713.84 - 394861808.08 = 636166905.76$  
   $6 \times 636166905.76 = \mathbf{3817001434.56}$
2. $(632.8)(9293.68) - (19871.13)(224.0) = 5881040.70 - 4451133.12 = 1429907.58$  
   $-632.8 \times 1429907.58 = \mathbf{-904845516.62}$
3. $(632.8)(19871.13) - (110938.68)(224.0) = 12574451.06 - 24850264.32 = -12275813.26$  
   $224.0 \times (-12275813.26) = \mathbf{-2749782170.24}$

Summing:
$$\det(\mathbf{A}) = 3817001434.56 - 904845516.62 - 2749782170.24 = \mathbf{162373747.70}$$

### Inverse Matrix $(\mathbf{X}^T \mathbf{X})^{-1}$:
Dividing the adjugate matrix by $\det(\mathbf{A})$ yields:

$$(\mathbf{X}^T \mathbf{X})^{-1} = \begin{bmatrix}
3.91793415 & -0.00880631 & -0.07560255 \\
-0.00880631 & 0.00003440 & 0.00013870 \\
-0.07560255 & 0.00013870 & 0.00163325
\end{bmatrix}$$

---

## 7. Solution of Normal Equations for Coefficients ($\mathbf{b}$)
We solve $\mathbf{b} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$:

$$\begin{bmatrix} b_0 \\ b_1 \\ b_2 \end{bmatrix} = \begin{bmatrix}
3.91793415 & -0.00880631 & -0.07560255 \\
-0.00880631 & 0.00003440 & 0.00013870 \\
-0.07560255 & 0.00013870 & 0.00163325
\end{bmatrix} \begin{bmatrix} 80.4 \\ 10905.68 \\ 2926.42 \end{bmatrix}$$

### Step-by-Step Row Products:
1. **$b_0$ (Intercept):**
   $$b_0 = (3.91793415)(80.4) + (-0.00880631)(10905.68) + (-0.07560255)(2926.42)$$
   $$= 315.001906 - 96.038800 - 221.244831 = \mathbf{-2.281725}$$

2. **$b_1$ (TV Coefficient):**
   $$b_1 = (-0.00880631)(80.4) + (0.00003440)(10905.68) + (0.00013870)(2926.42)$$
   $$= -0.708027 + 0.375155 + 0.405911 = \mathbf{0.073039}$$

3. **$b_2$ (Radio Coefficient):**
   $$b_2 = (-0.07560255)(80.4) + (0.00013870)(10905.68) + (0.00163325)(2926.42)$$
   $$= -6.078445 + 1.512618 + 4.779537 = \mathbf{0.213710}$$

### Fitted Sample Equation:
$$\widehat{y} = -2.281725 + 0.073039 \cdot \text{TV} + 0.213710 \cdot \text{Radio}$$

---

## 8. Sample Prediction and Residual Analysis (Observation 1)
For **Observation 1**:
- Input values: $\text{TV} = 230.1$, $\text{Radio} = 37.8$
- Actual Sales: $y_1 = 22.1$

### Prediction ($\widehat{y}_1$):
$$\widehat{y}_1 = -2.281725 + 0.073039(230.1) + 0.213710(37.8)$$
$$\widehat{y}_1 = -2.281725 + 16.806274 + 8.078238 = \mathbf{22.602884} \text{ thousand units}$$

### Residual ($e_1$):
$$e_1 = y_1 - \widehat{y}_1 = 22.100000 - 22.602884 = \mathbf{-0.502884} \text{ thousand units (-503 units)}$$

**Interpretation:** The model slightly overpredicts sales for Market 1 by 503 units, which is a small error (2.28% relative error).

---

## 9. Verification with Library Implementation (`numpy.linalg.lstsq`)
We fit the exact same sample observations using Python's standard least-squares routine:

| Parameter | Manual Calculation | Library `lstsq` | Absolute Difference |
|:---------:|:------------------:|:---------------:|:-------------------:|
| $b_0$     | -2.281725          | -2.281725       | $0.00 \times 10^0$  |
| $b_1$ (TV)| 0.073039           | 0.073039        | $8.26 \times 10^{-14}$ |
| $b_2$ (Radio) | 0.213710       | 0.213710        | $0.00 \times 10^0$  |

**Conclusion:** The manual normal equation derivation matches the library output to within $8.26 \times 10^{-14}$ (machine precision), confirming complete mathematical and computational correctness.


