# Chapter 7: Machine Learning Development Life Cycle (MLDLC)

## 📖 Table of Contents
- [Introduction](#introduction)
- [Overview of MLDLC](#overview-of-mldlc)
- [Phase 1: Frame the Problem](#phase-1-frame-the-problem)
- [Phase 2: Gather Data](#phase-2-gather-data)
- [Phase 3: Data Preprocessing](#phase-3-data-preprocessing)
- [Phase 4: Exploratory Data Analysis](#phase-4-exploratory-data-analysis-eda)
- [Phase 5: Feature Engineering](#phase-5-feature-engineering--selection)
- [Phase 6: Model Training & Evaluation](#phase-6-model-training-evaluation--selection)
- [Phase 7: Model Deployment](#phase-7-model-deployment)
- [Phase 8: Monitoring & Maintenance](#phase-8-monitoring--maintenance)
- [Best Practices](#best-practices)

---

## Introduction

Just as software development follows the SDLC (Software Development Life Cycle), machine learning projects follow the **MLDLC** (Machine Learning Development Life Cycle). However, ML projects have unique challenges:

### SDLC vs MLDLC

| Aspect | SDLC | MLDLC |
|--------|------|-------|
| **Primary Focus** | Code logic | Data + Model + Code |
| **Changes** | Code updates | Data, features, hyperparameters, architecture |
| **Testing** | Unit, integration tests | Performance metrics, data validation |
| **Deployment** | Deploy once, stable | Continuous retraining needed |
| **Maintenance** | Bug fixes, features | Model drift, data drift, retraining |
| **Predictability** | Deterministic | Probabilistic, uncertain |

**[PLACEHOLDER FOR SDLC VS MLDLC COMPARISON]**  
*Create two flowcharts side-by-side:*
- *Left: Traditional SDLC (Requirements → Design → Code → Test → Deploy)*
- *Right: MLDLC (showing all 8 phases with feedback loops)*
- *Highlight key differences*

---

## Overview of MLDLC

### The 8 Phases

```
1. Frame the Problem
   └─> Define objectives, success metrics
   
2. Gather Data
   └─> Collect from various sources
   
3. Data Preprocessing
   └─> Clean, transform, prepare
   
4. Exploratory Data Analysis (EDA)
   └─> Understand patterns, relationships
   
5. Feature Engineering & Selection
   └─> Create and select best features
   
6. Model Training, Evaluation & Selection
   └─> Build, compare, tune models
   
7. Model Deployment
   └─> Put model in production
   
8. Monitoring & Maintenance
   └─> Track performance, retrain
   └─> (Loop back to phase 1 or 2)
```

**[PLACEHOLDER FOR COMPLETE MLDLC DIAGRAM]**  
*Create a circular diagram showing:*
- *All 8 phases connected in cycle*
- *Icons for each phase*
- *Arrows showing flow and feedback loops*
- *Highlight iterative nature*
- *Color code by effort (time spent on each)*

---

## Phase 1: Frame the Problem

### Goal
**Define what you're trying to achieve and whether ML is the right solution.**

### Key Questions

#### 1.1 What is the Business Objective?

```
Examples:
❌ Bad: "We want to use AI"
✓ Good: "Reduce customer churn by 20% in 6 months"

❌ Bad: "Predict house prices"
✓ Good: "Help customers find fairly-priced homes within 10% accuracy"
```

#### 1.2 Is ML the Right Solution?

**Use this checklist**:

```
✓ Do we have data (or can we collect it)?
✓ Is there a pattern to learn (not purely random)?
✓ Is the problem too complex for simple rules?
✓ Can we tolerate some errors (not 100% accuracy required)?
✓ Will ML provide more value than cost?

If all ✓: ML might be appropriate
If any ✗: Reconsider or modify approach
```

#### 1.3 What Type of ML Problem?

```
Decision Tree:

Output is number? 
├─ Yes → Regression
└─ No → Output is category?
    ├─ Yes → Have labels?
    │   ├─ Yes → Classification (Supervised)
    │   └─ No → Clustering (Unsupervised)
    └─ No → Sequential decisions?
        └─ Yes → Reinforcement Learning
```

#### 1.4 Define Success Metrics

**How will you measure success?**

```
Examples:
- Regression: RMSE < $10,000
- Classification: Accuracy > 95%
- Business: Increase revenue by 15%
- User: Reduce time-to-decision by 30%
```

### Deliverables

```
✓ Problem statement document
✓ Success metrics defined
✓ ML approach identified
✓ Initial feasibility assessment
✓ Project timeline estimate
✓ Resource requirements
```

---

## Phase 2: Gather Data

### Goal
**Collect all relevant data needed to train and evaluate the model.**

### Data Sources

#### 2.1 Internal Sources
- Databases (SQL, NoSQL)
- Logs and events
- CRM systems
- Web analytics
- IoT sensors
- Past records

#### 2.2 External Sources
- Public datasets (Kaggle, UCI, government)
- APIs (Twitter, weather, maps)
- Web scraping
- Purchase data from vendors
- Partnerships

#### 2.3 Synthetic Data
- Simulations
- Data augmentation
- GANs (Generative Adversarial Networks)

**[PLACEHOLDER FOR DATA SOURCES DIAGRAM]**  
*Create an infographic showing:*
- *Center: ML Model*
- *Multiple arrows from different sources pointing to center*
- *Label each source with examples and icons*

---

### Data Collection Checklist

```
✓ Identify all potential data sources
✓ Assess data availability and accessibility
✓ Check data quality and completeness
✓ Verify legal/privacy compliance (GDPR, HIPAA)
✓ Estimate data collection costs and time
✓ Plan data storage infrastructure
✓ Set up data pipeline
✓ Document data sources and collection methods
```

### How Much Data?

**Rule of thumb**:

```
Simple Models:
- Linear Regression: 100-1,000 samples
- Logistic Regression: 1,000-10,000 samples

Medium Complexity:
- Random Forest: 10,000-100,000 samples
- Gradient Boosting: 10,000-100,000 samples

Deep Learning:
- Simple CNN: 50,000+ samples
- Complex models: 1,000,000+ samples

BUT: Quality > Quantity
1,000 good samples > 10,000 poor samples
```

### Deliverables

```
✓ Raw dataset collected
✓ Data dictionary (what each field means)
✓ Data source documentation
✓ Legal compliance verification
✓ Initial data quality report
```

---

## Phase 3: Data Preprocessing

### Goal
**Transform raw data into clean, usable format for modeling.**

This phase typically takes **60-80% of project time**!

### Steps in Data Preprocessing

#### 3.1 Data Cleaning

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('raw_data.csv')

# 1. Handle missing values
print("Missing values:\n", df.isnull().sum())

# Strategy depends on context:
df = df.dropna()  # Remove rows with missing values
# OR
df = df.fillna(df.mean())  # Fill with mean

# 2. Remove duplicates
df = df.drop_duplicates()

# 3. Fix data types
df['date'] = pd.to_datetime(df['date'])
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# 4. Handle outliers
# Using IQR method
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['price'] >= Q1 - 1.5*IQR) & (df['price'] <= Q3 + 1.5*IQR)]

# 5. Standardize text
df['name'] = df['name'].str.lower().str.strip()
df['category'] = df['category'].str.replace('_', ' ')
```

---

#### 3.2 Data Transformation

```python
# 1. Encoding categorical variables

# One-Hot Encoding (for nominal categories)
df = pd.get_dummies(df, columns=['color', 'brand'])

# Label Encoding (for ordinal categories)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['size'] = le.fit_transform(df['size'])  # S, M, L → 0, 1, 2

# 2. Feature Scaling

# Standardization (mean=0, std=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['age', 'income']] = scaler.fit_transform(df[['age', 'income']])

# Normalization (scale to 0-1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['age', 'income']] = scaler.fit_transform(df[['age', 'income']])

# 3. Handle imbalanced classes
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
```

---

#### 3.3 Train-Test Split

```python
from sklearn.model_selection import train_test_split

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# For time series: Don't shuffle!
# Use temporal split
split_date = '2023-01-01'
train = df[df['date'] < split_date]
test = df[df['date'] >= split_date]
```

**[PLACEHOLDER FOR TRAIN-TEST SPLIT VISUALIZATION]**  
*Show data being split:*
- *Original data (100%)*
- *Split into: Training (80%) and Test (20%)*
- *Show validation set creation from training*

---

### Deliverables

```
✓ Clean dataset
✓ Preprocessing pipeline (code)
✓ Transformations documented
✓ Train/test/validation sets
✓ Data statistics report
```

---

## Phase 4: Exploratory Data Analysis (EDA)

### Goal
**Understand your data through visualization and statistics.**

> "The greatest value of a picture is when it forces us to notice what we never expected to see." - John Tukey

### Why EDA?

- Understand data distributions
- Identify patterns and relationships
- Spot anomalies and outliers
- Validate assumptions
- Generate hypotheses
- Guide feature engineering

---

### EDA Techniques

#### 4.1 Descriptive Statistics

```python
import pandas as pd

# Basic statistics
print(df.describe())  # Mean, std, min, max, quartiles

# Distribution
print(df['age'].value_counts())

# Correlation
print(df.corr())
```

#### 4.2 Univariate Analysis

**Analyze each feature individually**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# For numerical features
plt.figure(figsize=(12, 4))

# Histogram
plt.subplot(131)
plt.hist(df['age'], bins=30)
plt.title('Age Distribution')

# Box plot
plt.subplot(132)
plt.boxplot(df['income'])
plt.title('Income (with outliers)')

# Density plot
plt.subplot(133)
df['age'].plot(kind='density')
plt.title('Age Density')

plt.tight_layout()
plt.show()
```

```python
# For categorical features
plt.figure(figsize=(10, 5))

# Bar chart
df['category'].value_counts().plot(kind='bar')
plt.title('Category Distribution')
plt.show()

# Pie chart
df['segment'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Customer Segments')
plt.show()
```

**[PLACEHOLDER FOR UNIVARIATE PLOTS]**  
*Create a grid showing:*
- *Histogram for continuous variable*
- *Box plot showing quartiles and outliers*
- *Bar chart for categorical variable*
- *Density plot*

---

#### 4.3 Bivariate Analysis

**Analyze relationships between two features**

```python
# Scatter plot (numerical vs numerical)
plt.scatter(df['age'], df['income'])
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Age vs Income')
plt.show()

# Box plot (categorical vs numerical)
sns.boxplot(x='category', y='price', data=df)
plt.title('Price by Category')
plt.show()

# Heatmap (correlation matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlations')
plt.show()

# Pair plot (all numerical features)
sns.pairplot(df[['age', 'income', 'spending', 'score']])
plt.show()
```

**[PLACEHOLDER FOR BIVARIATE PLOTS]**  
*Create examples:*
- *Scatter plot with trend line*
- *Correlation heatmap*
- *Grouped bar chart*

---

#### 4.4 Multivariate Analysis

```python
# 3D scatter plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['age'], df['income'], df['spending'])
ax.set_xlabel('Age')
ax.set_ylabel('Income')
ax.set_zlabel('Spending')
plt.show()

# Parallel coordinates
from pandas.plotting import parallel_coordinates
parallel_coordinates(df, 'category', colormap='viridis')
plt.show()
```

---

### Key Questions to Answer in EDA

```
Data Quality:
✓ Any missing values? How many?
✓ Any outliers? Are they errors or valid?
✓ Any duplicates?
✓ Correct data types?

Distribution:
✓ How is each feature distributed?
✓ Any skewness?
✓ Normal or non-normal distribution?

Relationships:
✓ Which features correlate with target?
✓ Which features correlate with each other?
✓ Any non-linear relationships?

Insights:
✓ Any surprising patterns?
✓ Any obvious groups/clusters?
✓ Any trends over time?
```

### Deliverables

```
✓ EDA report with visualizations
✓ Data insights document
✓ Feature correlation analysis
✓ Identified data quality issues
✓ Recommendations for feature engineering
```

---

## Phase 5: Feature Engineering & Selection

### Goal
**Create and select the most informative features for modeling.**

> "Feature engineering is the key to unlock the predictive power in data." - Pedro Domingos

### 5.1 Feature Creation

#### Creating New Features

```python
# Date/Time features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['quarter'] = df['date'].dt.quarter

# Mathematical transformations
df['log_price'] = np.log(df['price'] + 1)
df['sqrt_area'] = np.sqrt(df['area'])
df['price_per_sqft'] = df['price'] / df['sqft']

# Interaction features
df['age_income_interaction'] = df['age'] * df['income']
df['bedroom_bathroom_ratio'] = df['bedrooms'] / (df['bathrooms'] + 1)

# Binning continuous variables
df['age_group'] = pd.cut(df['age'], 
                         bins=[0, 18, 35, 50, 65, 100],
                         labels=['<18', '18-35', '35-50', '50-65', '65+'])

# Aggregations
customer_stats = df.groupby('customer_id').agg({
    'purchase_amount': ['sum', 'mean', 'count'],
    'days_since_purchase': 'min'
})

# Text features
df['text_length'] = df['description'].str.len()
df['word_count'] = df['description'].str.split().str.len()
df['has_discount_mention'] = df['description'].str.contains('discount|sale').astype(int)
```

---

### 5.2 Feature Selection

**Not all features are useful. Remove redundant/irrelevant features.**

#### Why Feature Selection?

- ✅ Reduce overfitting
- ✅ Improve model performance
- ✅ Reduce training time
- ✅ Improve interpretability
- ✅ Reduce storage

#### Methods

**Method 1: Filter Methods (Fast)**

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Select top 10 features based on ANOVA F-value
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()]
print("Selected features:", selected_features)

# Get feature scores
scores = pd.DataFrame({
    'feature': X.columns,
    'score': selector.scores_
}).sort_values('score', ascending=False)
```

**Method 2: Wrapper Methods (Thorough)**

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Recursive Feature Elimination
model = RandomForestClassifier()
rfe = RFE(estimator=model, n_features_to_select=10, step=1)
rfe.fit(X, y)

# Selected features
selected_features = X.columns[rfe.support_]
```

**Method 3: Embedded Methods (Efficient)**

```python
from sklearn.ensemble import RandomForestClassifier

# Train model with feature importance
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Get feature importances
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Keep only important features (threshold)
important_features = importances[importances['importance'] > 0.01]['feature']
X_selected = X[important_features]
```

**[PLACEHOLDER FOR FEATURE IMPORTANCE CHART]**  
*Bar chart showing:*
- *Features on Y-axis*
- *Importance scores on X-axis*
- *Color gradient: High (green) to Low (red)*
- *Threshold line*

---

### Deliverables

```
✓ Engineered features documented
✓ Feature selection report
✓ Final feature set
✓ Feature importance rankings
✓ Feature engineering pipeline (code)
```

---

## Phase 6: Model Training, Evaluation & Selection

### Goal
**Build, train, evaluate, and select the best performing model.**

### 6.1 Model Selection Strategy

```
Start Simple → Add Complexity as Needed

Level 1: Baseline
- Mean/Median (regression)
- Most frequent class (classification)
- Simple rules

Level 2: Linear Models
- Linear/Logistic Regression
- Naive Bayes

Level 3: Tree-Based
- Decision Tree
- Random Forest
- Gradient Boosting (XGBoost, LightGBM)

Level 4: Deep Learning
- Neural Networks
- CNN, RNN, Transformers

Progress only if previous level insufficient!
```

---

### 6.2 Model Training

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Try multiple models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier()
}

results = {}

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    
    results[name] = {
        'train_score': train_score,
        'test_score': test_score,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
# Compare results
results_df = pd.DataFrame(results).T
print(results_df)
```

---

### 6.3 Hyperparameter Tuning

**Find optimal model parameters**

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search
grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Use best model
best_model = grid_search.best_estimator_
```

**[PLACEHOLDER FOR HYPERPARAMETER TUNING VISUALIZATION]**  
*Heat map showing:*
- *Grid of hyperparameter combinations*
- *Color indicating performance*
- *Best combination highlighted*

---

### 6.4 Model Evaluation

**Use appropriate metrics for your problem**

#### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

#### Regression Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
```

---

### 6.5 Ensemble Learning

**Combine multiple models for better performance**

```python
from sklearn.ensemble import VotingClassifier, StackingClassifier

# Voting Classifier (simple averaging)
voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('xgb', XGBClassifier())
    ],
    voting='soft'  # Use probability averaging
)

voting_clf.fit(X_train, y_train)

# Stacking (use one model to combine others)
base_models = [
    ('rf', RandomForestClassifier()),
    ('xgb', XGBClassifier())
]

stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression()
)

stacking_clf.fit(X_train, y_train)
```

### Deliverables

```
✓ Trained models
✓ Model comparison report
✓ Best model selected
✓ Performance metrics documented
✓ Model artifacts saved
```

---

## Phase 7: Model Deployment

### Goal
**Put the model into production to serve predictions.**

### 7.1 Save the Model

```python
import pickle
import joblib

# Method 1: Pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Method 2: Joblib (better for scikit-learn)
joblib.dump(model, 'model.joblib')

# Load model
model = joblib.load('model.joblib')
```

---

### 7.2 Create Prediction Pipeline

```python
# Save entire pipeline
from sklearn.pipeline import Pipeline

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=10)),
    ('model', RandomForestClassifier())
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Save pipeline
joblib.dump(pipeline, 'pipeline.joblib')

# Later: Load and predict
pipeline = joblib.load('pipeline.joblib')
prediction = pipeline.predict(new_data)
```

---

### 7.3 Create API for Model

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model at startup
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        
        # Preprocess
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0].tolist()
        
        # Return response
        return jsonify({
            'prediction': int(prediction),
            'probability': probability,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

### 7.4 Deployment Options

**Option 1: Cloud Platforms**
- AWS SageMaker
- Google Cloud AI Platform
- Azure ML
- Heroku, PythonAnywhere

**Option 2: Containers**
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model.joblib scaler.joblib app.py .

EXPOSE 5000

CMD ["python", "app.py"]
```

**Option 3: Serverless**
- AWS Lambda
- Google Cloud Functions
- Azure Functions

**[PLACEHOLDER FOR DEPLOYMENT ARCHITECTURE]**  
*Diagram showing:*
- *Client → API Gateway → Model Server → Database*
- *Load Balancer for scaling*
- *Monitoring service*

---

### Deliverables

```
✓ Deployed model (production)
✓ API endpoints
✓ Deployment documentation
✓ Health check endpoints
✓ Rollback plan
```

---

## Phase 8: Monitoring & Maintenance

### Goal
**Ensure model continues to perform well in production.**

### 8.1 What to Monitor

#### Performance Metrics
```python
# Track over time
metrics_to_monitor = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'latency': [],  # Prediction time
    'throughput': [],  # Requests/second
    'error_rate': []
}
```

#### Data Drift
```python
# Monitor input distribution changes
from scipy.stats import ks_2samp

# Compare training vs production data
statistic, p_value = ks_2samp(train_feature, production_feature)

if p_value < 0.05:
    alert("Data drift detected!")
```

#### Concept Drift
```
Monitor: Relationship between features and target changes

Example: 
- Trained model: "Summer = High ice cream sales"
- New pattern: Climate change → Unpredictable seasons
- Model performance degrades
```

---

### 8.2 Monitoring Dashboard

```python
# Example: Log predictions
import logging
from datetime import datetime

def log_prediction(features, prediction, probability, actual=None):
    log_entry = {
        'timestamp': datetime.now(),
        'features': features.tolist(),
        'prediction': prediction,
        'probability': probability,
        'actual': actual  # If available later
    }
    
    # Save to database
    save_to_db(log_entry)
    
    # Calculate metrics if actual available
    if actual is not None:
        update_metrics(prediction, actual)
```

---

### 8.3 Retraining Strategy

**When to Retrain?**

```
Triggers:
✓ Scheduled (weekly, monthly)
✓ Performance drops below threshold
✓ Significant data drift detected
✓ New data accumulated (e.g., 10k new samples)
✓ Business requirements change
```

**Retraining Pipeline**:

```python
def retrain_pipeline():
    # 1. Collect new data
    new_data = fetch_recent_data(days=30)
    
    # 2. Combine with historical data (optional)
    combined_data = combine_datasets(historical_data, new_data)
    
    # 3. Preprocess
    processed_data = preprocess(combined_data)
    
    # 4. Train new model
    new_model = train_model(processed_data)
    
    # 5. Evaluate
    if evaluate_model(new_model) > current_threshold:
        # 6. Deploy new model
        deploy_model(new_model, version='v2')
        
        # 7. Keep old model for rollback
        archive_model(old_model, version='v1')
    else:
        alert("New model underperforms. Keeping current model.")
```

---

### 8.4 A/B Testing

**Test new model before full deployment**

```
Strategy:
1. Deploy new model to 10% of traffic
2. Compare performance with current model (90% traffic)
3. If new model better: Gradually increase traffic
4. If worse: Rollback
```

---

### 8.5 Model Backup & Versioning

```python
# Version every model
model_version = {
    'version': 'v1.2.3',
    'date': '2024-10-23',
    'accuracy': 0.95,
    'data_range': '2023-01-01 to 2024-10-01',
    'features': feature_list,
    'hyperparameters': params
}

# Save with version
joblib.dump(model, f'model_{model_version["version"]}.joblib')
save_metadata(model_version)
```

---

### Deliverables

```
✓ Monitoring dashboard
✓ Alerting system
✓ Retraining schedule
✓ Performance reports
✓ Model versioning system
✓ Rollback procedures
```

---

## Best Practices

### 1. Documentation

```
Document everything:
✓ Problem definition and goals
✓ Data sources and collection methods
✓ Preprocessing steps and decisions
✓ Feature engineering logic
✓ Model selection rationale
✓ Hyperparameter tuning process
✓ Deployment architecture
✓ Monitoring setup
```

### 2. Version Control

```
✓ Code (Git)
✓ Data (DVC, Git LFS)
✓ Models (MLflow, Weights & Biases)
✓ Experiments (MLflow, TensorBoard)
```

### 3. Reproducibility

```python
# Set random seeds
import random
import numpy as np
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Document environment
pip freeze > requirements.txt
```

### 4. Testing

```python
# Unit tests for preprocessing
def test_preprocessing():
    input_data = create_test_data()
    output_data = preprocess(input_data)
    assert output_data.shape == expected_shape
    assert output_data.isnull().sum() == 0

# Integration tests for pipeline
def test_pipeline():
    sample_input = create_sample()
    prediction = pipeline.predict(sample_input)
    assert 0 <= prediction <= 1  # For probability
```

### 5. Collaboration

```
Use tools:
- Jupyter Notebooks (exploration)
- Git (code)
- MLflow/Weights&Biases (experiments)
- Confluence/Notion (documentation)
- Slack/Teams (communication)
```

---

## Summary

🎯 **Key Takeaways**:

**The 8 Phases of MLDLC**:
1. **Frame Problem** - Define goals, metrics, approach
2. **Gather Data** - Collect from multiple sources
3. **Preprocess** - Clean, transform, split (60-80% of time!)
4. **EDA** - Understand data through visualization
5. **Feature Engineering** - Create and select best features
6. **Model Training** - Build, tune, evaluate models
7. **Deployment** - Put model in production
8. **Monitor & Maintain** - Track performance, retrain

**Remember**: MLDLC is **iterative**, not linear. You'll loop back multiple times!

**Time Distribution** (typical project):
- Data Collection & Preprocessing: 50-60%
- EDA & Feature Engineering: 20-25%
- Modeling: 10-15%
- Deployment & Monitoring: 10-15%

---

*Previous: [← Challenges in ML](./06_challenges_in_ml.md)*  
*Next: [Job Roles in ML →](./08_job_roles_in_ml.md)*
