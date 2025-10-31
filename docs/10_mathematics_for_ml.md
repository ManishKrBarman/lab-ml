# Chapter 10: Essential Mathematics for Machine Learning

## 📖 Table of Contents
- [Introduction](#introduction)
- [Linear Algebra](#linear-algebra)
- [Calculus](#calculus)
- [Probability and Statistics](#probability-and-statistics)
- [Optimization](#optimization)

---

## Introduction

Machine learning is fundamentally built on mathematics. While you don't need a PhD in math to use ML, understanding the basics will help you:
- Debug models effectively
- Tune hyperparameters intelligently
- Understand what's happening "under the hood"
- Read research papers and documentation

> "Mathematics is the language of machine learning."

**What You Need to Know**:
```
Essential (Must Know):
├─ Linear Algebra: Vectors, matrices, operations
├─ Basic Calculus: Derivatives, gradients
├─ Statistics: Mean, variance, distributions
└─ Probability: Basic probability, conditional probability

Advanced (Good to Know):
├─ Multivariable Calculus
├─ Matrix Calculus
├─ Information Theory
└─ Optimization Theory
```

---

## Linear Algebra

### Why Linear Algebra Matters

**Every ML algorithm uses linear algebra:**
- Data is stored in vectors/matrices
- Neural network weights are matrices
- Transformations are matrix multiplications
- Dimensionality reduction uses matrix factorization

---

### Scalars, Vectors, and Matrices

#### Scalars
Single numbers: `x = 5`, `learning_rate = 0.01`

#### Vectors
1D arrays of numbers:
```python
import numpy as np

# Column vector (default in math)
v = np.array([[1], [2], [3]])

# Row vector
v_row = np.array([1, 2, 3])

# Common operations
magnitude = np.linalg.norm(v)  # Length of vector
print(magnitude)  # 3.742
```

**Vector Operations**:
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Addition
print(a + b)  # [5, 7, 9]

# Scalar multiplication
print(2 * a)  # [2, 4, 6]

# Dot product
print(np.dot(a, b))  # 1*4 + 2*5 + 3*6 = 32

# Element-wise multiplication
print(a * b)  # [4, 10, 18]
```

**[PLACEHOLDER FOR VECTOR OPERATIONS VISUAL]**  
*Show:*
- *Vector addition as arrow addition*
- *Scalar multiplication as scaling*
- *Dot product geometric interpretation*

---

#### Matrices
2D arrays:
```python
# 2x3 matrix
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(A.shape)  # (2, 3) - 2 rows, 3 columns
```

**Matrix Operations**:

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Addition
print(A + B)
# [[6, 8], [10, 12]]

# Scalar multiplication
print(3 * A)
# [[3, 6], [9, 12]]

# Matrix multiplication
print(A @ B)  # or np.dot(A, B)
# [[19, 22], [43, 50]]

# Transpose
print(A.T)
# [[1, 3], [2, 4]]

# Determinant (square matrix only)
print(np.linalg.det(A))  # -2.0

# Inverse (if exists)
print(np.linalg.inv(A))
# [[-2. , 1. ], [1.5, -0.5]]
```

---

### Key Concepts

#### Identity Matrix
```python
I = np.eye(3)  # 3x3 identity matrix
# [[1, 0, 0],
#  [0, 1, 0],
#  [0, 0, 1]]

# Property: A @ I = I @ A = A
```

#### Matrix-Vector Multiplication
```python
# Used in neural networks constantly!
A = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2
x = np.array([2, 3])                     # 2x1

result = A @ x  # Shape: (3,)
print(result)   # [8, 18, 28]

# How it works:
# [1*2 + 2*3] = [8]
# [3*2 + 4*3] = [18]
# [5*2 + 6*3] = [28]
```

**In Neural Networks**:
```python
# Forward pass in a layer
W = np.random.rand(128, 64)   # Weights: 64 inputs → 128 outputs
x = np.random.rand(64)        # Input vector
b = np.random.rand(128)       # Bias

output = W @ x + b  # Matrix-vector multiplication!
# Shape: (128,) - 128 outputs
```

---

### Eigenvalues and Eigenvectors

**Definition**: For matrix A, if `Av = λv`, then:
- `v` is an eigenvector
- `λ` is an eigenvalue

```python
A = np.array([[4, 2], [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)    # [5. 2.]
print("Eigenvectors:\n", eigenvectors)
```

**Why They Matter**:
- **PCA**: Uses eigenvectors to find principal components
- **PageRank**: Dominant eigenvector
- **Stability Analysis**: In dynamical systems

---

### ML Applications

#### Linear Regression
```python
# y = Xw + b
# Find w that minimizes error

# Normal equation: w = (X^T X)^(-1) X^T y
X = np.array([[1, 1], [1, 2], [1, 3]])  # Feature matrix
y = np.array([2, 4, 6])                  # Targets

# Add bias term
X_with_bias = np.c_[np.ones(3), X]

# Solve using normal equation
w = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
print(w)  # Optimal weights
```

#### Neural Network Forward Pass
```python
# Each layer: output = activation(W @ input + b)
W1 = np.random.rand(128, 784)  # Layer 1: 784 → 128
W2 = np.random.rand(10, 128)   # Layer 2: 128 → 10

x = np.random.rand(784)        # Input (e.g., 28x28 image flattened)

# Forward pass
h1 = np.maximum(0, W1 @ x)     # ReLU activation
output = W2 @ h1               # Output layer
```

**[PLACEHOLDER FOR LINEAR ALGEBRA IN ML]**  
*Show:*
- *Data matrix X (samples × features)*
- *Weight matrix W*
- *Matrix multiplication X @ W*
- *Result as transformed space*

---

## Calculus

### Why Calculus Matters

**Gradient descent** - the core of ML optimization - uses calculus!

```
Training a Model:
1. Make prediction: ŷ = model(X)
2. Calculate loss: L = (y - ŷ)²
3. Compute gradient: ∂L/∂w  ← CALCULUS!
4. Update weights: w = w - α * ∂L/∂w
5. Repeat
```

---

### Derivatives

**Derivative = Rate of change**

```python
# Example: f(x) = x²
# Derivative: f'(x) = 2x

# At x=3: f'(3) = 6
# Meaning: At x=3, f is increasing at rate 6
```

**Common Derivatives**:
```
f(x) = c        → f'(x) = 0
f(x) = x        → f'(x) = 1
f(x) = x²       → f'(x) = 2x
f(x) = x³       → f'(x) = 3x²
f(x) = eˣ       → f'(x) = eˣ
f(x) = ln(x)    → f'(x) = 1/x
f(x) = sin(x)   → f'(x) = cos(x)
```

**Python Implementation**:
```python
import numpy as np

# Numerical derivative
def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

# Example
f = lambda x: x**2
print(derivative(f, 3))  # ≈ 6.0 (exact: 2*3 = 6)
```

---

### Chain Rule

**For composite functions**: If `y = f(g(x))`, then:
```
dy/dx = (dy/dg) * (dg/dx)
```

**Example**:
```python
# y = (2x + 1)³
# Let g = 2x + 1, then y = g³

# dy/dg = 3g²
# dg/dx = 2
# dy/dx = 3g² * 2 = 6(2x + 1)²
```

**Why It Matters**: **Backpropagation uses chain rule!**

```python
# Neural network with 2 layers
# y = W2 @ σ(W1 @ x)
#     ↑     ↑   ↑
#     Layer2 Activation Layer1

# Chain rule for gradient:
# ∂L/∂W1 = (∂L/∂y) * (∂y/∂σ) * (∂σ/∂W1)
```

**[PLACEHOLDER FOR CHAIN RULE VISUAL]**  
*Show:*
- *Computational graph with nodes*
- *Forward pass arrows*
- *Backward pass (gradient flow) arrows*
- *Chain rule application*

---

### Partial Derivatives

For functions of multiple variables: `f(x, y, z)`

**Partial derivative** = derivative with respect to one variable, keeping others fixed

```python
# f(x, y) = x² + 3xy + y²

# ∂f/∂x = 2x + 3y  (treat y as constant)
# ∂f/∂y = 3x + 2y  (treat x as constant)

import numpy as np

def f(x, y):
    return x**2 + 3*x*y + y**2

# Numerical partial derivatives
def partial_x(x, y, h=1e-5):
    return (f(x + h, y) - f(x - h, y)) / (2 * h)

def partial_y(x, y, h=1e-5):
    return (f(x, y + h) - f(x, y - h)) / (2 * h)

print(partial_x(2, 3))  # ≈ 13.0 (exact: 2*2 + 3*3 = 13)
print(partial_y(2, 3))  # ≈ 12.0 (exact: 3*2 + 2*3 = 12)
```

---

### Gradients

**Gradient** = vector of all partial derivatives

For `f(x, y, z)`:
```
∇f = [∂f/∂x, ∂f/∂y, ∂f/∂z]
```

**Gradient points in direction of steepest increase!**

```python
# Loss function: L(w1, w2) = (w1 - 3)² + (w2 - 4)²
# Minimum at (3, 4)

def loss(w):
    return (w[0] - 3)**2 + (w[1] - 4)**2

def gradient(w):
    return np.array([
        2*(w[0] - 3),  # ∂L/∂w1
        2*(w[1] - 4)   # ∂L/∂w2
    ])

# Gradient descent
w = np.array([0.0, 0.0])  # Start
alpha = 0.1               # Learning rate

for i in range(100):
    grad = gradient(w)
    w = w - alpha * grad  # Move opposite to gradient!

print(w)  # ≈ [3, 4] - found the minimum!
```

**[PLACEHOLDER FOR GRADIENT DESCENT VISUAL]**  
*Show:*
- *3D loss surface (bowl shape)*
- *Gradient vectors pointing uphill*
- *Optimization path moving downhill*
- *Minimum point at bottom*

---

### ML Applications

#### Backpropagation
```python
# Simple neural network gradient computation
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Forward pass
x = np.array([1.0, 2.0])     # Input
W1 = np.array([[0.5, 0.5],   # Layer 1 weights
               [0.5, 0.5]])
W2 = np.array([0.5, 0.5])     # Layer 2 weights

z1 = W1 @ x                   # Linear transformation
a1 = sigmoid(z1)              # Activation
z2 = W2 @ a1                  # Output layer
a2 = sigmoid(z2)              # Output

y_true = 1.0                  # Target
loss = (y_true - a2)**2       # MSE loss

# Backward pass (chain rule!)
dL_da2 = -2 * (y_true - a2)
da2_dz2 = sigmoid_derivative(z2)
dz2_dW2 = a1
dL_dW2 = dL_da2 * da2_dz2 * dz2_dW2  # Gradient for W2

print("Gradient for W2:", dL_dW2)
```

---

## Probability and Statistics

### Why Probability Matters

Machine learning is fundamentally about:
- **Uncertainty**: Data is noisy
- **Generalization**: Probability of correct prediction
- **Bayesian ML**: Prior beliefs + data → posterior beliefs

---

### Basic Probability

**Probability**: P(A) = likelihood of event A

```python
import numpy as np

# Coin flip
outcomes = ['H', 'T']
P_heads = 0.5
P_tails = 0.5

# Die roll
P_six = 1/6

# Simula probability
n_trials = 10000
coin_flips = np.random.choice(['H', 'T'], size=n_trials)
estimated_P_heads = np.sum(coin_flips == 'H') / n_trials
print(f"Estimated P(Heads): {estimated_P_heads:.3f}")  # ≈ 0.500
```

---

### Conditional Probability

**P(A|B)** = Probability of A given B happened

```
P(A|B) = P(A ∩ B) / P(B)
```

**Example**: Email spam detection
```python
# P(spam | contains "winner")
# = P(contains "winner" | spam) * P(spam) / P(contains "winner")
# This is Bayes' Theorem!
```

---

### Distributions

#### Normal Distribution (Gaussian)

Most important distribution in ML!

```python
import matplotlib.pyplot as plt
from scipy import stats

# Normal distribution: N(μ, σ²)
mu = 0      # Mean
sigma = 1   # Standard deviation

# Generate samples
samples = np.random.normal(mu, sigma, size=10000)

# Plot
plt.hist(samples, bins=50, density=True, alpha=0.7)
x = np.linspace(-4, 4, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2)
plt.title('Normal Distribution N(0, 1)')
plt.show()
```

**Properties**:
- 68% of data within 1 standard deviation
- 95% within 2 standard deviations
- 99.7% within 3 standard deviations

**[PLACEHOLDER FOR NORMAL DISTRIBUTION]**  
*Show:*
- *Bell curve*
- *Mark μ, μ±σ, μ±2σ, μ±3σ*
- *Shade 68%, 95%, 99.7% areas*

---

#### Other Important Distributions

**Bernoulli**: Binary outcome (0 or 1)
```python
# Coin flip
p = 0.5  # Probability of 1
samples = np.random.binomial(1, p, size=1000)
```

**Categorical**: Multiple outcomes
```python
# Die roll
probabilities = [1/6] * 6
samples = np.random.choice(range(1, 7), size=1000, p=probabilities)
```

**Uniform**: All outcomes equally likely
```python
# Random number in [0, 1)
samples = np.random.uniform(0, 1, size=1000)
```

---

### Statistical Measures

```python
data = np.array([2, 4, 4, 4, 5, 5, 7, 9])

# Central Tendency
mean = np.mean(data)         # 5.0 (average)
median = np.median(data)     # 4.5 (middle value)
from scipy import stats
mode = stats.mode(data)[0]   # 4 (most frequent)

# Spread
variance = np.var(data)      # 4.0 (average squared deviation)
std_dev = np.std(data)       # 2.0 (√variance)
range_val = np.ptp(data)     # 7 (max - min)

print(f"Mean: {mean}, Median: {median}, Std Dev: {std_dev}")
```

---

### Bayes' Theorem

**Formula**:
```
P(A|B) = P(B|A) * P(A) / P(B)
```

**ML Application**: Naive Bayes Classifier

```python
from sklearn.naive_bayes import GaussianNB

# Email spam classification
X = [[1, 0], [0, 1], [1, 1], [0, 0]]  # Features: [has_winner, has_free]
y = [1, 0, 1, 0]                       # Labels: 1=spam, 0=not spam

model = GaussianNB()
model.fit(X, y)

# Predict
new_email = [[1, 1]]  # Contains both "winner" and "free"
prediction = model.predict(new_email)
probability = model.predict_proba(new_email)

print(f"Spam probability: {probability[0][1]:.2f}")
```

---

### ML Applications

#### Maximum Likelihood Estimation (MLE)
```python
# Estimate parameters from data
# Find parameters that maximize P(data | parameters)

data = np.random.normal(5, 2, size=1000)  # True: μ=5, σ=2

# MLE estimates (sample statistics)
mu_estimate = np.mean(data)     # ≈ 5
sigma_estimate = np.std(data)   # ≈ 2

print(f"Estimated μ: {mu_estimate:.2f}, σ: {sigma_estimate:.2f}")
```

#### Confidence Intervals
```python
from scipy import stats

# 95% confidence interval for mean
confidence = 0.95
n = len(data)
mean = np.mean(data)
sem = stats.sem(data)  # Standard error of mean

confidence_interval = stats.t.interval(confidence, n-1, loc=mean, scale=sem)
print(f"95% CI: ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f})")
```

---

## Optimization

### Gradient Descent

**Goal**: Minimize loss function L(w)

**Algorithm**:
```
1. Initialize weights w randomly
2. Repeat:
   - Compute gradient: g = ∇L(w)
   - Update: w = w - α * g
3. Until convergence
```

**Implementation**:
```python
def gradient_descent(gradient_fn, initial_w, learning_rate=0.01, iterations=1000):
    w = initial_w
    history = [w]
    
    for i in range(iterations):
        gradient = gradient_fn(w)
        w = w - learning_rate * gradient
        history.append(w)
    
    return w, history

# Example: Minimize f(x) = x²
def gradient(x):
    return 2 * x  # f'(x) = 2x

optimal_x, history = gradient_descent(gradient, initial_w=10.0)
print(f"Optimal x: {optimal_x:.6f}")  # ≈ 0 (minimum of x²)
```

---

### Variants of Gradient Descent

#### Batch Gradient Descent
```python
# Use all data for each update
for epoch in range(n_epochs):
    gradient = compute_gradient(X, y, weights)  # All data
    weights = weights - learning_rate * gradient
```

#### Stochastic Gradient Descent (SGD)
```python
# Use one sample at a time
for epoch in range(n_epochs):
    for i in range(len(X)):
        gradient = compute_gradient(X[i], y[i], weights)  # One sample
        weights = weights - learning_rate * gradient
```

#### Mini-batch Gradient Descent
```python
# Use small batches
batch_size = 32
for epoch in range(n_epochs):
    for batch in get_batches(X, y, batch_size):
        gradient = compute_gradient(batch_X, batch_y, weights)
        weights = weights - learning_rate * gradient
```

---

### Advanced Optimizers

**Momentum**:
```python
# Accelerates SGD by adding fraction of previous update
velocity = 0
for iteration in range(n_iterations):
    gradient = compute_gradient(X, y, weights)
    velocity = momentum * velocity - learning_rate * gradient
    weights = weights + velocity
```

**Adam** (most popular):
```python
# Combines momentum + adaptive learning rate
# Implemented in all ML frameworks

import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(n_epochs):
    optimizer.zero_grad()
    loss = compute_loss(X, y)
    loss.backward()
    optimizer.step()  # Adam update
```

---

## Summary

🎯 **Key Takeaways**:

**Linear Algebra**:
- Vectors and matrices represent data and weights
- Matrix multiplication = transformations
- Eigenvalues/vectors used in PCA

**Calculus**:
- Derivatives measure rate of change
- Gradients point to steepest increase
- Chain rule enables backpropagation
- Optimization uses gradients to minimize loss

**Probability & Statistics**:
- Probability quantifies uncertainty
- Distributions model data
- Bayes' theorem for updating beliefs
- Statistics for model evaluation

**Optimization**:
- Gradient descent minimizes loss
- Various optimizers (SGD, Adam, etc.)
- Learning rate is crucial hyperparameter

**Remember**: You don't need to memorize formulas - understand concepts and let libraries handle computation!

---

*Previous: [← Understanding Tensors](./09_tensors.md)*  
*Next: [ML Frameworks & Libraries →](./11_ml_frameworks.md)*
