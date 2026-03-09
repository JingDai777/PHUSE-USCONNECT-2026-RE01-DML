# Robust Causal Inference in Real-world Evidence Studies with Double Machine Learning

[![PHUSE 2026](https://img.shields.io/badge/Conference-PHUSE%202026-blue)](https://phuse.global/)

## 📌 Overview
This repository contains the python code, and conceptual framework for the **PHUSE 2026** paper: *"Robust Causal Inference in Real-world Evidence Studies with Double Machine Learning"*

As clinical research increasingly shifts toward Large-Scale Real-World Data (RWD), traditional propensity score methods often struggle with high-dimensional confounders and non-linear relationships. This paper introduces **Double Machine Learning (DML)** as a modern alternative that provides valid statistical inference even when complex black-box models are used to estimate nuisance parameters.

---

## 🧠 What is Double Machine Learning (DML)?



Double Machine Learning (Chernozhukov et al., 2018) is a framework that uses "orthogonalization" to remove the bias introduced by high-dimensional confounders. 

### The Core Logic:
1.  **Stage 1 (Residualizing the Outcome):** Predict the outcome ($Y$) from the covariates ($X$) and calculate the residuals ($Y - \hat{Y}$).
2.  **Stage 2 (Residualizing the Treatment):** Predict the treatment ($T$) from the covariates ($X$)—**this is essentially the ML Propensity Score**—and calculate the residuals ($T - \hat{T}$).
3.  **Stage 3 (Estimation):** Regress the outcome residuals on the treatment residuals. Because the influence of $X$ has been "scrubbed" from both variables, the resulting coefficient is the true causal effect.

---

## 🔗 DML and Propensity Score Methods

DML does not discard Propensity Scores; it **incorporates them into a "Double-Robust" framework.**

* **Propensity Score (PS) Methods:** Focus primarily on the treatment assignment mechanism $P(T|X)$. If the PS model is misspecified (e.g., a simple logistic regression fails to capture complex interactions), the resulting treatment effect will be biased.
* **DML Advantage:** DML models *both* the treatment (Propensity) and the outcome (Prognostic). By using **Neyman-Orthogonality**, it ensures that even if one of these models has a slight estimation error (common in ML), the error doesn't "leak" into the final treatment effect estimate.

---

## ⚖️ Model Selection Strategy: DML vs PS Methods

| Feature | IPTW / PSM | DML |
| :--- | :--- | :--- |
| **Primary Goal** | Covariate balance across groups.| Bias reduction in high-dimensional and nonlinear settings.|
| **Model Focus** | Mostly treatment assignment.| Both treatment and outcome models.|
| **Complexity** |Low; easy to explain to clinicians.| High; requires careful explanation.|
| **Best Use Case** | Low-dimensional, well-understood data.| High-dimensional data with complex interactions.|

### ✅ Use Propensity Methods When:
* The number of covariates is small ($p < 20$).
* The relationship between confounders and treatment is well-understood and likely linear.
* You need a "Table 1" to demonstrate covariate balance to a regulatory body.

### 🚀 Use Double Machine Learning When:
* You are dealing with hundreds or thousands of covariates (e.g., omics data or unstructured EHR).
* The "assignment" of a drug is complex and depends on non-linear thresholds (e.g., age, weight, and lab values interacting).
* You want to use high-performance models (XGBoost, Lasso, Neural Nets) but still need **valid P-values and Confidence Intervals.**

---

## 🛠 Python Implementation (EconML)

This project utilizes Microsoft’s `EconML` library to implement the DML framework.

```python
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 1. Define the models (Nuisance Learners)
# model_y: Predicts outcome (Prognostic Model)
# model_t: Predicts treatment (Propensity Model)
est = LinearDML(
    model_y=RandomForestRegressor(n_estimators=100, max_depth=5),
    model_t=RandomForestClassifier(n_estimators=100, max_depth=5),
    discrete_treatment=True,
    cv=5  # Cross-fitting folds
)

# 2. Fit the model
# Y = Outcome, T = Treatment, X = Confounders
est.fit(Y, T, X=X, W=None)

# 3. Get the Treatment Effect
te_estimate = est.effect(X_test)
print(f"Estimated Treatment Effect: {est.ate_}")
print(est.summary())
