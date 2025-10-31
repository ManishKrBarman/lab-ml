# 🎯 Quick Reference Guide

A quick lookup guide for key concepts, commands, and resources from the ML Learning Path.

## 📑 Table of Contents
- [Key Concepts](#key-concepts)
- [Common Commands](#common-commands)
- [Important Formulas](#important-formulas)
- [Library Imports](#library-imports)
- [Model Selection](#model-selection)
- [Troubleshooting](#troubleshooting)

---

## Key Concepts

### ML Types Quick Lookup

| Type | Labeled Data? | Use Case | Example Algorithms |
|------|--------------|----------|-------------------|
| **Supervised** | Yes | Predict outcomes | Linear Regression, Random Forest, Neural Networks |
| **Unsupervised** | No | Find patterns | K-Means, PCA, DBSCAN |
| **Semi-Supervised** | Partial | Limited labels | Self-training, Co-training |
| **Reinforcement** | Rewards | Sequential decisions | Q-Learning, DQN, PPO |

### Learning Approaches

| Approach | How It Works | Speed | Example |
|----------|--------------|-------|---------|
| **Instance-Based** | Remember examples | Fast training, slow prediction | KNN |
| **Model-Based** | Learn patterns | Slow training, fast prediction | Linear Regression |

### Training Strategies

| Strategy | Data Usage | Update Frequency | Best For |
|----------|-----------|------------------|----------|
| **Batch** | All at once | Infrequent | Stable data |
| **Online** | One at a time | Continuous | Streaming data |
| **Mini-Batch** | Small batches | Regular | Deep learning |

---

## Common Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv myenv

# Activate (Windows)
myenv\Scripts\activate

# Activate (Mac/Linux)
source myenv/bin/activate

# Install packages
pip install numpy pandas scikit-learn matplotlib jupyter

# Save dependencies
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt

# Deactivate
deactivate
```

### Conda Commands
```bash
# Create environment
conda create -n ml_env python=3.11

# Activate
conda activate ml_env

# Install packages
conda install numpy pandas scikit-learn matplotlib jupyter

# List environments
conda env list

# Export environment
conda env export > environment.yml

# Create from file
conda env create -f environment.yml

# Deactivate
conda deactivate

# Remove environment
conda env remove -n ml_env
```

### Jupyter
```bash
# Start Jupyter Notebook
jupyter notebook

# Start Jupyter Lab
jupyter lab

# Install kernel
python -m ipykernel install --user --name=myenv
```

### Git Basics
```bash
# Initialize repository
git init

# Add files
git add .

# Commit
git commit -m "Message"

# Push to GitHub
git remote add origin https://github.com/username/repo.git
git push -u origin main

# Clone repository
git clone https://github.com/username/repo.git
```

---

## Important Formulas

### Evaluation Metrics

**Accuracy**:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision**:
```
Precision = TP / (TP + FP)
```

**Recall**:
```
Recall = TP / (TP + FN)
```

**F1 Score**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Mean Squared Error**:
```
MSE = (1/n) × Σ(y_true - y_pred)²
```

**R² Score**:
```
R² = 1 - (SS_res / SS_tot)
```

### Gradient Descent
```
w_new = w_old - α × ∂L/∂w

where:
- α = learning rate
- ∂L/∂w = gradient of loss w.r.t. weights
```

---

## Library Imports

### Standard ML Workflow
```python
# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Evaluation
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             mean_squared_error, r2_score)

# Model selection
from sklearn.model_selection import cross_val_score, GridSearchCV
```

### Deep Learning (TensorFlow)
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
```

### Deep Learning (PyTorch)
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
```

---

## Model Selection

### When to Use Which Algorithm?

#### Classification

**Logistic Regression**
- ✅ Binary classification
- ✅ Need interpretability
- ✅ Linear decision boundary
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
```

**Decision Tree**
- ✅ Non-linear relationships
- ✅ Need interpretability
- ✅ Mixed data types
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5)
```

**Random Forest**
- ✅ Non-linear relationships
- ✅ Robust to overfitting
- ✅ Feature importance needed
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
```

**XGBoost**
- ✅ Tabular data
- ✅ Best performance needed
- ✅ Kaggle competitions
```python
import xgboost as xgb
model = xgb.XGBClassifier()
```

**Neural Network**
- ✅ Complex patterns
- ✅ Large dataset
- ✅ Images, text, or sequences
```python
model = keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
```

#### Regression

**Linear Regression**
- ✅ Linear relationship
- ✅ Simple baseline
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
```

**Random Forest Regressor**
- ✅ Non-linear relationships
- ✅ Robust predictions
```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100)
```

**XGBoost Regressor**
- ✅ Best performance on tabular data
```python
model = xgb.XGBRegressor()
```

#### Clustering

**K-Means**
- ✅ Known number of clusters
- ✅ Spherical clusters
```python
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
```

**DBSCAN**
- ✅ Unknown number of clusters
- ✅ Arbitrary shapes
- ✅ Outlier detection
```python
from sklearn.cluster import DBSCAN
model = DBSCAN(eps=0.5, min_samples=5)
```

---

## Troubleshooting

### Common Errors

**1. ImportError: No module named 'sklearn'**
```bash
# Solution
pip install scikit-learn
```

**2. ModuleNotFoundError**
```bash
# Check virtual environment is activated
which python  # Mac/Linux
where python  # Windows

# Should point to your venv, not system Python
```

**3. CUDA not available**
```python
# Check GPU
import torch
print(torch.cuda.is_available())

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**4. Kernel died (Jupyter)**
```bash
# Increase memory or reduce batch size
# Or restart kernel: Kernel → Restart
```

**5. Shape mismatch errors**
```python
# Always check shapes
print(X.shape, y.shape)

# Reshape if needed
X = X.reshape(-1, 1)  # For single feature
y = y.ravel()  # Flatten y
```

### Performance Issues

**Model Training Too Slow**
```python
# Reduce data
X_sample = X[:10000]

# Reduce model complexity
model = RandomForestClassifier(n_estimators=10)  # Instead of 100

# Use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

**Out of Memory**
```python
# Reduce batch size
batch_size = 16  # Instead of 32 or 64

# Use data generators
train_gen = tf.keras.preprocessing.image.ImageDataGenerator()

# Clear session (TensorFlow)
tf.keras.backend.clear_session()
```

### Model Not Learning

**Underfitting**
```python
# Increase model complexity
model = RandomForestClassifier(n_estimators=200, max_depth=20)

# Add more features
# Train longer
model.fit(X, y, epochs=100)  # Instead of 10
```

**Overfitting**
```python
# Reduce complexity
model = RandomForestClassifier(n_estimators=50, max_depth=10)

# Add regularization
model = LogisticRegression(C=0.1)  # Smaller C = more regularization

# Use dropout (Neural Networks)
model.add(layers.Dropout(0.5))

# Get more data
# Use data augmentation
```

---

## Quick Workflows

### Standard ML Pipeline
```python
# 1. Load data
import pandas as pd
df = pd.read_csv('data.csv')

# 2. Split features and target
X = df.drop('target', axis=1)
y = df['target']

# 3. Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Evaluate
from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# 7. Save model
import joblib
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

### Deep Learning Pipeline (TensorFlow)
```python
# 1. Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Build model
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# 3. Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
)

# 5. Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")

# 6. Save
model.save('my_model.h5')
```

---

## Useful Resources

### Documentation
- [NumPy Docs](https://numpy.org/doc/)
- [Pandas Docs](https://pandas.pydata.org/docs/)
- [Scikit-learn Docs](https://scikit-learn.org/stable/)
- [TensorFlow Docs](https://www.tensorflow.org/api_docs)
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)

### Practice Platforms
- [Kaggle](https://www.kaggle.com/) - Competitions and datasets
- [Google Colab](https://colab.research.google.com/) - Free GPU notebooks
- [HackerRank](https://www.hackerrank.com/domains/ai) - ML challenges

### Learning Resources
- [Fast.ai](https://www.fast.ai/) - Practical deep learning
- [Coursera](https://www.coursera.org/) - ML courses
- [Papers with Code](https://paperswithcode.com/) - Latest research

---

**Keep this guide handy for quick reference during your ML journey! 🚀**
