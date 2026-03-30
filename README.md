# Linear Regression from Scratch (Gradient Descent)

This project implements **Linear Regression using Gradient Descent from scratch** with NumPy.

No machine learning libraries (like scikit-learn models) are used for training — the goal is to understand the **core mechanics of regression and optimization**.

---

## Objective

Predict **Salary** based on **Years of Experience** using a linear model:

```
Salary = w * Experience + b
```

---

## Dataset

A real-world simple regression dataset:

* Feature: `YearsExperience`
* Target: `Salary`

---

## Key Idea

Instead of directly computing parameters, we use **Gradient Descent** to iteratively learn:

* `w` (slope)
* `b` (intercept)

---

## How the Model Works

### 1. Prediction

```
y_pred = Xw + b
```

---

### 2. Error

```
error = y_pred - y
```

---

### 3. Loss (MSE)

```
MSE = (1/n) * Σ(error²)
```

---

### 4. Gradients

```
dw = (1/n) * Xᵀ(error)
db = (1/n) * Σ(error)
```

---

### 5. Update Rule

```
w = w - lr * dw
b = b - lr * db
```

This process repeats for multiple iterations until the model converges.

---

## Project Structure

```
linear_regression/
│
├── notebooks/
│   └── linear_regression.ipynb   # experiment + visualization
│
├── src/
│   └── linear_regression.py      # core implementation
│
├── data/
│   └── Salary_Data.csv          # dataset
│
└── README.md
```

---

## Training Pipeline

1. Load dataset
2. Convert to NumPy arrays
3. Train-test split (80/20)
4. Train model using gradient descent
5. Predict on test data
6. Evaluate using MSE / RMSE
7. Visualize regression line

---

## Results

* Learned relationship:

```
Salary ≈ 9880 * Experience + 21912
```

* Test Performance:

```
MSE  ≈ 5.28 × 10⁷
RMSE ≈ 7,200
```

Interpretation:

* Average prediction error ≈ ₹7k
* Model captures linear trend effectively

---

## Visualization

* Blue → training data
* Orange → test data
* Red line → learned regression model

The model fits both training and unseen test data well, indicating **good generalization**.

---

## Important Note

* The **Linear Regression model is implemented entirely from scratch**
* `scikit-learn` is used **only for train-test splitting**, not for modeling

---

## Concepts Covered

* Linear Regression
* Gradient Descent
* Mean Squared Error (MSE)
* Train-Test Split
* Model Generalization
* Convergence behavior

---

## Why this Project Matters

This project demonstrates:

* Ability to implement ML algorithms from first principles
* Understanding of optimization (gradient descent)
* Proper ML workflow (train/test evaluation)
* Clean project structuring (src vs notebook)

---

## Author

Ashish Pise
