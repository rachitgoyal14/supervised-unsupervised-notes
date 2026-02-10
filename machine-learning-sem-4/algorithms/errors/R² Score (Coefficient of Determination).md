## Motivation: The Baseline Problem

### Scenario: University Advertisement

You need to advertise expected packages to incoming freshers, but you don't have their CGPA data yet.

| Student | CGPA | Package |
|---------|------|---------|
| 1       | ?    | 6 LPA   |
| 2       | ?    | 4 LPA   |
| 3       | ?    | 7 LPA   |
| 4       | ?    | 3 LPA   |
| 5       | ?    | 5 LPA   |

**Worst-case approach:** Use the **average package** (mean)
```
Average = (6 + 4 + 7 + 3 + 5) / 5 = 5 LPA
```

Tell all students: "Expected package is 5 LPA"

**Better approach:** Use **linear regression** to create a predictive model based on CGPA
- Draw best fit line
- Make personalized predictions based on each student's CGPA

**Question:** How much better is the regression model compared to just using the mean?

**Answer:** R² Score tells us this!

---

## Definition

**R² (R-squared)** measures how much better your model is compared to simply predicting the mean value every time.

Also called:
- **Coefficient of Determination**
- **Goodness of Fit**

---

## Mathematical Formula
```
R² = 1 - (SSR / SST)
```

Where:

**SSR (Sum of Squared Residuals):**
```
SSR = Σᵢ₌₁ⁿ (yᵢ - ŷᵢ)²
```
- Error of the **regression line**
- How far predictions are from actual values

**SST (Sum of Squared Total):**
```
SST = Σᵢ₌₁ⁿ (yᵢ - ȳ)²
```
- Error of the **mean line** (baseline)
- How far actual values are from the mean
- Also called **SSM** (Sum of Squares Mean)

**Full formula:**
```
R² = 1 - [Σ(yᵢ - ŷᵢ)²] / [Σ(yᵢ - ȳ)²]
```

---

## Interpretation

### Range of Values

**R² lies between -∞ and 1** (typically 0 to 1)

| R² Value | Meaning |
|----------|---------|
| **R² = 1** | Perfect prediction. Model explains 100% of variance. |
| **R² = 0.9** | Model explains 90% of variance. Very good! |
| **R² = 0.5** | Model explains 50% of variance. Moderate. |
| **R² = 0** | Model is no better than predicting the mean. |
| **R² < 0** | Model is **worse** than predicting the mean! |

---

### What Does Each Value Mean?

**R² = 1 (Perfect fit)**
- SSR = 0 (no residual error)
- All predictions exactly match actual values
- Regression line passes through all points

**R² = 0 (No improvement)**
- SSR = SST
- Regression model provides no better predictions than the mean
- Your model is useless

**R² < 0 (Worse than baseline)**
- SSR > SST
- Your model makes worse predictions than just using the mean
- Model is completely wrong or severely overfitted

---

## Variance Interpretation

### Single Feature Example
```
R² = 0.9 for CGPA → Package prediction
```

**Interpretation:**
"CGPA explains **90% of the variance** in Package"

**What this means:**
- 90% of variation in packages can be explained by CGPA
- 10% of variation is due to other factors (IQ, skills, luck, etc.)

---

### Multiple Features Example
```
Model: Package = β₀ + β₁(CGPA) + β₂(IQ)
R² = 0.97
```

**Interpretation:**
"CGPA and IQ together explain **97% of the variance** in Package"

---

## The Problem with R² in Multiple Regression

### Adding Features Always Increases R²

**Example:**

| Features                | R²                      |
| ----------------------- | ----------------------- |
| CGPA only               | 0.95                    |
| CGPA + IQ               | 0.97 (relevant feature) |
| CGPA + IQ + Temperature | 0.975 (irrelevant!)     |

**Problem:** Even adding **irrelevant features** (like temperature) increases or maintains R²!

**Why?**
- More features → more parameters to fit the training data
- Model can always find some spurious correlation
- Leads to overfitting

**What should happen:** R² should **decrease** when adding irrelevant features

**But it doesn't!** This is a major disadvantage of R².

---

## Adjusted R²

### Definition

**Adjusted R²** penalizes adding unnecessary features, solving the problem above.

### Formula
```
Adjusted R² = 1 - [(1 - R²)(n - 1) / (n - k - 1)]
```

Where:
- **R²** = regular R² score
- **n** = number of rows (samples)
- **k** = number of independent features (independent variables/input columns)

---

### How It Works

**Adding a relevant feature:**
- R² increases significantly
- (1 - R²) decreases significantly
- Penalty from (n-1)/(n-k-1) is small
- **Adjusted R² increases** 

**Adding an irrelevant feature:**
- R² increases slightly (or stays same)
- (1 - R²) decreases slightly
- Penalty from (n-1)/(n-k-1) increases (k increases)
- **Adjusted R² decreases or stays same** 

---

### Example Calculation

**Setup:**
- n = 100 samples
- Original model: CGPA only
- R² = 0.95, k = 1
```
Adjusted R² = 1 - [(1 - 0.95)(100 - 1) / (100 - 1 - 1)]
            = 1 - [0.05 × 99 / 98]
            = 1 - 0.0505
            = 0.9495
```

**Add IQ (relevant feature):**
- New R² = 0.97, k = 2
```
Adjusted R² = 1 - [(1 - 0.97)(100 - 1) / (100 - 2 - 1)]
            = 1 - [0.03 × 99 / 97]
            = 1 - 0.0306
            = 0.9694
```

**Adjusted R² increased from 0.9495 → 0.9694**  Good feature!

**Add Temperature (irrelevant feature):**
- New R² = 0.975 (slight increase), k = 3
```
Adjusted R² = 1 - [(1 - 0.975)(100 - 1) / (100 - 3 - 1)]
            = 1 - [0.025 × 99 / 96]
            = 1 - 0.0258
            = 0.9742
```

Wait, it still increased! But the increase is much smaller than expected.

Let's try when R² barely increases:
- New R² = 0.9705 (tiny increase), k = 3
```
Adjusted R² = 1 - [(1 - 0.9705)(100 - 1) / (100 - 3 - 1)]
            = 1 - [0.0295 × 99 / 96]
            = 1 - 0.0304
            = 0.9696
```

**Adjusted R² changed from 0.9694 → 0.9696** (barely any improvement)

This signals that the feature isn't helping much!

---

## When to Use Which

| Metric | Use Case |
|--------|----------|
| **R²** | Simple Linear Regression (1 feature) |
| **Adjusted R²** | Multiple Linear Regression (2+ features) |

**Key Difference:**
- **R²** always increases with more features
- **Adjusted R²** penalizes unnecessary features

---

## Preventing Overfitting

**Adjusted R²** helps prevent overfitting by:

1. **Penalizing model complexity** (more features = higher penalty)
2. **Identifying irrelevant features** (adjusted R² won't improve much)
3. **Encouraging simpler models** (fewer features with similar performance)

**Feature Selection Strategy:**
```
1. Start with no features
2. Add features one by one
3. Check if Adjusted R² improves
4. Keep feature if Adjusted R² increases significantly
5. Remove feature if Adjusted R² doesn't improve
```

---

## Comparison Table

| Aspect                   | R²                       | Adjusted R²               |
| ------------------------ | ------------------------ | ------------------------- |
| **Formula**              | 1 - (SSR/SST)            | 1 - [(1-R²)(n-1)/(n-k-1)] |
| **Range**                | 0 to 1 (can be negative) | 0 to 1 (can be negative)  |
| **Adding features**      | Always increases         | Can decrease              |
| **Use case**             | Simple regression        | Multiple regression       |
| **Prevents overfitting** | No                       | Yes                       |
| **Penalizes complexity** | No                       | Yes                       |

---

## Key Takeaways

1. **R² measures goodness of fit** (how much better than mean prediction)
2. **R² = 1 means perfect fit**, R² = 0 means no better than mean
3. **R² can be negative** if model is worse than baseline
4. **R² represents variance explained** by features
5. **R² always increases** with more features (problem!)
6. **Adjusted R² penalizes unnecessary features**
7. **Use Adjusted R² for multiple regression** to prevent overfitting
8. Both are **scale-independent** (can compare across problems)

---

## Python Implementation
```python
from sklearn.metrics import r2_score
import numpy as np

# Actual and predicted values
y_actual = np.array([3, 4, 5, 6, 7])
y_predicted = np.array([2.8, 4.2, 5.1, 5.9, 7.2])

# Calculate R²
r2 = r2_score(y_actual, y_predicted)
print(f"R² Score: {r2:.4f}")

# Manual calculation
y_mean = np.mean(y_actual)
sst = np.sum((y_actual - y_mean) ** 2)
ssr = np.sum((y_actual - y_predicted) ** 2)
r2_manual = 1 - (ssr / sst)
print(f"R² Manual: {r2_manual:.4f}")

# Calculate Adjusted R²
n = len(y_actual)  # number of samples
k = 1  # number of features
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
print(f"Adjusted R²: {adjusted_r2:.4f}")
```