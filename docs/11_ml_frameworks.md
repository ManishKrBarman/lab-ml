# Chapter 11: Machine Learning Frameworks and Libraries

## 📖 Table of Contents
- [Introduction](#introduction)
- [Core Python Libraries](#core-python-libraries)
- [Traditional ML Frameworks](#traditional-ml-frameworks)
- [Deep Learning Frameworks](#deep-learning-frameworks)
- [Specialized Libraries](#specialized-libraries)
- [Cloud Platforms](#cloud-platforms)
- [Framework Comparison](#framework-comparison)

---

## Introduction

The Python ecosystem offers powerful tools for machine learning, from data manipulation to model deployment. This chapter covers the essential platforms and software you'll use throughout your ML journey.

### The ML Stack

```
┌─────────────────────────────────────┐
│       Cloud Platforms               │  AWS, Azure, GCP
├─────────────────────────────────────┤
│       Deployment Tools              │  TensorFlow Serving, ONNX
├─────────────────────────────────────┤
│       Deep Learning Frameworks      │  TensorFlow, PyTorch, Keras
├─────────────────────────────────────┤
│       Traditional ML Libraries      │  Scikit-learn, XGBoost
├─────────────────────────────────────┤
│       Data Processing               │  Pandas, NumPy
├─────────────────────────────────────┤
│       Python                        │  Foundation
└─────────────────────────────────────┘
```

**[PLACEHOLDER FOR ML STACK DIAGRAM]**  
*Create a layered architecture diagram showing:*
- *Python at base*
- *Data libraries (NumPy, Pandas) above*
- *ML frameworks (scikit-learn, TensorFlow, PyTorch) next*
- *Cloud services at top*
- *Arrows showing data flow*

---

## Core Python Libraries

### NumPy - Numerical Computing

**What**: Foundation for numerical computing in Python

**Why**: Fast array operations, mathematical functions

**Installation**:
```bash
pip install numpy
```

**Key Features**:
```python
import numpy as np

# Array creation
arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
random = np.random.rand(2, 2)

# Operations
arr_squared = arr ** 2
mean = np.mean(arr)
sum_val = np.sum(arr)

# Linear algebra
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
product = A @ B  # Matrix multiplication

# Broadcasting
matrix = np.array([[1, 2, 3], [4, 5, 6]])
vector = np.array([10, 20, 30])
result = matrix + vector  # Broadcasting!
```

**Use Cases**:
- Numerical computations
- Array operations
- Linear algebra
- Random number generation
- Foundation for other libraries

---

### Pandas - Data Manipulation

**What**: Data manipulation and analysis library

**Why**: Easy-to-use data structures (DataFrame), powerful data processing

**Installation**:
```bash
pip install pandas
```

**Key Features**:
```python
import pandas as pd

# Create DataFrame
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
}
df = pd.DataFrame(data)

# Read data
df = pd.read_csv('data.csv')
df = pd.read_excel('data.xlsx')
df = pd.read_json('data.json')

# Data exploration
print(df.head())           # First 5 rows
print(df.info())           # Column info
print(df.describe())       # Statistics
print(df.shape)            # (rows, columns)

# Selection
df['age']                  # Column
df[df['age'] > 28]        # Filter rows
df.loc[0]                  # Row by label
df.iloc[0]                 # Row by position

# Operations
df['age_next_year'] = df['age'] + 1
df.groupby('age')['salary'].mean()
df.sort_values('salary', ascending=False)

# Handle missing data
df.dropna()                # Remove missing
df.fillna(0)              # Fill missing with 0

# Merge datasets
df_merged = pd.merge(df1, df2, on='id')
```

**Use Cases**:
- Data loading and cleaning
- Exploratory data analysis
- Feature engineering
- Data transformation

---

### Matplotlib & Seaborn - Visualization

**What**: Data visualization libraries

**Why**: Create plots, charts, and visual insights

**Installation**:
```bash
pip install matplotlib seaborn
```

**Matplotlib - Low-level plotting**:
```python
import matplotlib.pyplot as plt

# Line plot
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simple Plot')
plt.show()

# Multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot([1, 2, 3], [1, 4, 9])
axes[0, 1].scatter([1, 2, 3], [1, 4, 9])
axes[1, 0].hist([1, 2, 2, 3, 3, 3, 4])
axes[1, 1].bar(['A', 'B', 'C'], [10, 20, 15])
plt.tight_layout()
plt.show()
```

**Seaborn - High-level plotting**:
```python
import seaborn as sns

# Set style
sns.set_style('whitegrid')

# Distribution plot
sns.histplot(data=df, x='age', bins=20)

# Scatter plot with regression
sns.regplot(data=df, x='age', y='salary')

# Box plot
sns.boxplot(data=df, x='category', y='value')

# Correlation heatmap
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')

# Pair plot
sns.pairplot(df, hue='species')
```

**[PLACEHOLDER FOR VISUALIZATION EXAMPLES]**  
*Show grid of common plot types:*
- *Line plot, scatter plot, histogram, box plot*
- *Heatmap, pair plot, distribution plot*
- *Each labeled with use case*

---

## Traditional ML Frameworks

### Scikit-learn - ML Swiss Army Knife

**What**: Comprehensive machine learning library

**Why**: Easy-to-use, consistent API, covers most traditional ML algorithms

**Installation**:
```bash
pip install scikit-learn
```

**Key Features**:

#### Classification
```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Example: Random Forest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
```

#### Regression
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Example
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}, R²: {r2:.2f}")
```

#### Clustering
```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
```

#### Dimensionality Reduction
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(f"Explained variance: {pca.explained_variance_ratio_}")

# t-SNE (for visualization)
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X)
```

#### Preprocessing
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Standardization (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler!

# Normalization (range 0-1)
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Encode categorical variables
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
```

#### Model Selection
```python
from sklearn.model_selection import cross_val_score, GridSearchCV

# Cross-validation
scores = cross_val_score(model, X, y, cv=5)
print(f"CV Scores: {scores}, Mean: {scores.mean():.2f}")

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.2f}")
```

#### Pipelines
```python
from sklearn.pipeline import Pipeline

# Create pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Predict (all steps applied automatically!)
y_pred = pipeline.predict(X_test)
```

**Use Cases**:
- Traditional ML algorithms
- Data preprocessing
- Model selection and evaluation
- Production-ready pipelines

**[PLACEHOLDER FOR SCIKIT-LEARN WORKFLOW]**  
*Show flowchart:*
- *Data → Preprocessing → Model Selection → Training*
- *→ Evaluation → Hyperparameter Tuning → Deployment*
- *Show scikit-learn tools at each step*

---

### XGBoost - Gradient Boosting

**What**: Optimized gradient boosting library

**Why**: State-of-the-art performance for tabular data, wins Kaggle competitions

**Installation**:
```bash
pip install xgboost
```

**Usage**:
```python
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Classification
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Feature importance
import matplotlib.pyplot as plt
xgb.plot_importance(model)
plt.show()

# Regression
model_reg = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)
model_reg.fit(X_train, y_train)
```

**Alternatives**:
- **LightGBM**: Faster, more memory-efficient
- **CatBoost**: Handles categorical features automatically

---

## Deep Learning Frameworks

### TensorFlow & Keras

**What**: End-to-end deep learning platform

**Why**: Production-ready, extensive ecosystem, TensorFlow Serving for deployment

**Installation**:
```bash
pip install tensorflow
```

**Keras (High-level API)**:
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Sequential model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")

# Predict
predictions = model.predict(X_test)
```

**Functional API (More flexible)**:
```python
# For complex architectures
inputs = keras.Input(shape=(784,))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

**CNN Example**:
```python
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

**RNN Example**:
```python
model = keras.Sequential([
    layers.Embedding(vocab_size, 64),
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(64),
    layers.Dense(10, activation='softmax')
])
```

**Transfer Learning**:
```python
# Use pre-trained model
base_model = keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Add custom layers
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
```

**Custom Training Loop**:
```python
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
for epoch in range(epochs):
    for x_batch, y_batch in train_dataset:
        loss = train_step(x_batch, y_batch)
```

**Use Cases**:
- Production deployments
- Mobile/edge devices (TensorFlow Lite)
- Serving models at scale
- Research and prototyping

---

### PyTorch

**What**: Research-focused deep learning framework

**Why**: Pythonic, dynamic computation graphs, popular in research

**Installation**:
```bash
pip install torch torchvision
```

**Basic Usage**:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Create model
model = SimpleNet()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
```

**CNN in PyTorch**:
```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 5 * 5)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

**Transfer Learning**:
```python
import torchvision.models as models

# Load pre-trained model
model = models.resnet50(pretrained=True)

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Only train final layer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

**GPU Support**:
```python
# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move model and data to GPU
model = model.to(device)
data = data.to(device)
target = target.to(device)
```

**Use Cases**:
- Research and experimentation
- Custom architectures
- Dynamic computation graphs
- NLP (popular with Transformers)

---

### TensorFlow vs PyTorch

| Feature | TensorFlow/Keras | PyTorch |
|---------|------------------|---------|
| **Ease of Use** | Very easy (Keras) | Moderate |
| **Flexibility** | Good | Excellent |
| **Deployment** | Excellent (TF Serving) | Good (TorchServe) |
| **Mobile** | TensorFlow Lite | PyTorch Mobile |
| **Community** | Large | Large (growing) |
| **Industry** | Production-focused | Research-focused |
| **Debugging** | Harder (static graphs) | Easier (dynamic) |
| **API Style** | High-level (Keras) | Pythonic |

**Recommendation**:
- **TensorFlow/Keras**: Production, deployment, mobile
- **PyTorch**: Research, experimentation, custom models

**[PLACEHOLDER FOR TF VS PYTORCH COMPARISON]**  
*Create side-by-side comparison visual:*
- *Code snippets for same task in both*
- *Highlight differences in API style*
- *Show deployment options*

---

## Specialized Libraries

### Computer Vision

**OpenCV**:
```bash
pip install opencv-python
```

```python
import cv2

# Load image
img = cv2.imread('image.jpg')

# Resize
resized = cv2.resize(img, (224, 224))

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Edge detection
edges = cv2.Canny(gray, 100, 200)

# Display
cv2.imshow('Image', img)
cv2.waitKey(0)
```

**Pillow**:
```bash
pip install Pillow
```

```python
from PIL import Image

img = Image.open('image.jpg')
img_resized = img.resize((224, 224))
img_rotated = img.rotate(45)
img.save('output.jpg')
```

---

### Natural Language Processing

**NLTK**:
```bash
pip install nltk
```

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text = "Machine learning is amazing!"
tokens = word_tokenize(text)
stop_words = set(stopwords.words('english'))
filtered = [w for w in tokens if w.lower() not in stop_words]
```

**spaCy**:
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

```python
import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# Named Entity Recognition
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")
```

**Transformers (Hugging Face)**:
```bash
pip install transformers
```

```python
from transformers import pipeline

# Sentiment analysis
classifier = pipeline('sentiment-analysis')
result = classifier("I love machine learning!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]

# Text generation
generator = pipeline('text-generation', model='gpt2')
result = generator("Machine learning is", max_length=50)
```

---

### AutoML

**AutoKeras**:
```bash
pip install autokeras
```

```python
import autokeras as ak

# Automated image classification
clf = ak.ImageClassifier(max_trials=10)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
```

**H2O AutoML**:
```bash
pip install h2o
```

```python
import h2o
from h2o.automl import H2OAutoML

h2o.init()
train = h2o.import_file('train.csv')
test = h2o.import_file('test.csv')

aml = H2OAutoML(max_runtime_secs=3600)
aml.train(y='target', training_frame=train)

# Get best model
best_model = aml.leader
predictions = best_model.predict(test)
```

---

## Cloud Platforms

### AWS (Amazon Web Services)

**SageMaker**:
```python
import sagemaker
from sagemaker import get_execution_role

role = get_execution_role()

# Train model
from sagemaker.estimator import Estimator

estimator = Estimator(
    image_uri='your-algorithm',
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge'
)

estimator.fit({'training': 's3://bucket/data'})

# Deploy
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium'
)
```

---

### Google Cloud Platform

**Vertex AI**:
```python
from google.cloud import aiplatform

aiplatform.init(project='your-project', location='us-central1')

# Train model
job = aiplatform.CustomTrainingJob(
    display_name='training-job',
    script_path='train.py',
    container_uri='gcr.io/cloud-aiplatform/training/tf-cpu.2-8:latest'
)

model = job.run(
    dataset=dataset,
    model_display_name='my-model'
)
```

---

### Azure

**Azure ML**:
```python
from azureml.core import Workspace, Experiment, Environment
from azureml.train.sklearn import SKLearn

ws = Workspace.from_config()

# Create experiment
experiment = Experiment(workspace=ws, name='my-experiment')

# Train
estimator = SKLearn(
    source_directory='./src',
    entry_script='train.py',
    compute_target='cpu-cluster',
    environment_definition=Environment.from_conda_specification(
        name='sklearn-env',
        file_path='environment.yml'
    )
)

run = experiment.submit(estimator)
```

---

## Framework Comparison

### Quick Reference

| Task | Recommended Library |
|------|---------------------|
| **Data Manipulation** | Pandas |
| **Numerical Computing** | NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Traditional ML** | Scikit-learn |
| **Gradient Boosting** | XGBoost, LightGBM |
| **Deep Learning (Production)** | TensorFlow/Keras |
| **Deep Learning (Research)** | PyTorch |
| **Computer Vision** | OpenCV, torchvision |
| **NLP** | Transformers, spaCy |
| **AutoML** | AutoKeras, H2O |

---

### Typical Workflow

```python
# 1. Data Loading & Exploration
import pandas as pd
df = pd.read_csv('data.csv')

# 2. Preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Traditional ML
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# OR Deep Learning
import tensorflow as tf
model = tf.keras.Sequential([...])
model.compile(...)
model.fit(X_train, y_train)

# 4. Evaluation
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

# 5. Deployment
model.save('model.pkl')  # Scikit-learn
model.save('model.h5')   # Keras
```

---

## Summary

🎯 **Key Takeaways**:

**Essential Libraries**:
- **NumPy**: Numerical computing foundation
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Visualization
- **Scikit-learn**: Traditional ML algorithms
- **TensorFlow/PyTorch**: Deep learning

**Choose Your Tools Based On**:
- **Task complexity**: Simple → scikit-learn, Complex → Deep learning
- **Data type**: Tabular → XGBoost, Images/Text → Neural networks
- **Deployment**: Production → TensorFlow, Research → PyTorch
- **Time budget**: Quick → AutoML, Custom → Manual implementation

**Learning Path**:
1. Master NumPy & Pandas (data handling)
2. Learn Scikit-learn (traditional ML)
3. Pick one deep learning framework (TensorFlow OR PyTorch)
4. Explore specialized libraries as needed

**Remember**: Start simple, use the right tool for the job, and don't over-engineer!

---

*Previous: [← Mathematics for ML](./10_mathematics_for_ml.md)*  
*Next: [Development Environment Setup →](./12_setup_guide.md)*
