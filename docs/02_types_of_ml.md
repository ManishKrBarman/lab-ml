# Chapter 2: Types of Machine Learning

## 📖 Table of Contents
- [Overview](#overview)
- [Supervised Learning](#supervised-learning)
- [Unsupervised Learning](#unsupervised-learning)
- [Semi-Supervised Learning](#semi-supervised-learning)
- [Reinforcement Learning](#reinforcement-learning)
- [Comparison and When to Use Each](#comparison-and-when-to-use-each)

---

## Overview

Machine Learning algorithms are categorized based on **how they learn** from data. Understanding these categories is crucial for choosing the right approach for your problem.

### The Four Main Types

```
Machine Learning Types
    │
    ├── Supervised Learning (Learning with a teacher)
    │   ├── Regression
    │   └── Classification
    │
    ├── Unsupervised Learning (Learning without a teacher)
    │   ├── Clustering
    │   ├── Dimensionality Reduction
    │   ├── Anomaly Detection
    │   └── Association Rule Learning
    │
    ├── Semi-Supervised Learning (Mix of both)
    │
    └── Reinforcement Learning (Learning by trial and error)
```

**[PLACEHOLDER FOR HIERARCHICAL DIAGRAM]**  
*Create a tree diagram showing:*
- *Root: "Machine Learning"*
- *Four main branches with icons (teacher for supervised, explorer for unsupervised, mix icon for semi-supervised, game controller for reinforcement)*
- *Sub-branches showing specific techniques*
- *Use different colors for each main category*

---

## Supervised Learning

### What is Supervised Learning?

**Supervised Learning** is like learning with a teacher. You're given:
- **Input data (X)**: Features/attributes
- **Output labels (Y)**: Correct answers
- **Goal**: Learn the relationship between X and Y to predict Y for new, unseen X

### Real-World Analogy
Imagine learning math with a teacher:
- Teacher gives you problems (X) with solutions (Y)
- You study the examples and learn the pattern
- Later, you solve new problems on your own
- The teacher checks if you're correct

### How It Works

```python
# Training Phase
Input: Historical data with labels
[House Size, Location, Age] → [Price]
[1500 sq ft, Downtown, 10 years] → $450,000
[1200 sq ft, Suburb, 5 years] → $320,000
...
Model learns the relationship

# Prediction Phase
Input: New house without price
[1800 sq ft, Downtown, 8 years] → Model predicts: $490,000
```

**[PLACEHOLDER FOR SUPERVISED LEARNING DIAGRAM]**  
*Create a flowchart showing:*
- *Top: Training data (input features + labels) going into "Model Training"*
- *Middle: Model box with gears/brain icon*
- *Bottom: New input → Model → Predicted output*
- *Use arrows to show data flow*
- *Include visual examples (house images → price tags)*

---

## Types of Supervised Learning

### 1. Regression (Numerical Output)

Predicting **continuous numerical values**.

#### Characteristics
- Output is a **number** on a continuous scale
- Can take any value within a range
- Examples: prices, temperatures, heights, sales

#### Common Algorithms
- Linear Regression
- Polynomial Regression
- Support Vector Regression (SVR)
- Decision Tree Regression
- Random Forest Regression
- Neural Networks

#### Real Examples

| Problem | Input Features | Output (Continuous) |
|---------|---------------|---------------------|
| **House Price Prediction** | Size, location, rooms, age | $450,000 |
| **Temperature Forecasting** | Historical temp, humidity, pressure | 28.5°C |
| **Stock Price Prediction** | Historical prices, volume, indicators | $156.73 |
| **Salary Estimation** | Experience, education, role, location | $85,000 |
| **Crop Yield Prediction** | Rainfall, fertilizer, soil quality | 4.2 tons/acre |

#### Code Example: Linear Regression

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Training data: Years of experience vs Salary
X_train = np.array([[1], [2], [3], [4], [5]])  # Experience
y_train = np.array([40000, 45000, 55000, 60000, 70000])  # Salary

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict salary for someone with 6 years experience
new_experience = np.array([[6]])
predicted_salary = model.predict(new_experience)
print(f"Predicted salary: ${predicted_salary[0]:,.0f}")
# Output: Predicted salary: $75,000
```

**[PLACEHOLDER FOR REGRESSION GRAPH]**  
*Create a scatter plot with line of best fit:*
- *X-axis: Years of Experience (0-10)*
- *Y-axis: Salary ($)*
- *Blue dots: Training data points*
- *Red line: Regression line*
- *Green star: New prediction point*
- *Title: "Linear Regression: Salary vs Experience"*

---

### 2. Classification (Categorical Output)

Predicting **discrete categories or classes**.

#### Characteristics
- Output is a **category/label**
- Limited set of possible outcomes
- Can be binary (2 classes) or multi-class (3+ classes)

#### Common Algorithms
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Decision Trees
- Random Forest
- Naive Bayes
- Neural Networks

#### Real Examples

| Problem | Input Features | Output (Category) | Type |
|---------|---------------|-------------------|------|
| **Email Spam Detection** | Email content, sender, links | Spam / Not Spam | Binary |
| **Disease Diagnosis** | Symptoms, test results, age | Healthy / Diseased | Binary |
| **Loan Approval** | Income, credit score, debt | Approved / Rejected | Binary |
| **Handwritten Digit Recognition** | Pixel values | 0, 1, 2, ..., 9 | Multi-class |
| **Flower Species Classification** | Petal length, width, color | Setosa / Versicolor / Virginica | Multi-class |
| **Customer Churn Prediction** | Usage, complaints, tenure | Will Churn / Won't Churn | Binary |

#### Code Example: Binary Classification

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Training data: Study hours vs Pass/Fail
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])  # Hours studied
y = np.array([0, 0, 0, 1, 1, 1, 1, 1])  # 0=Fail, 1=Pass

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict for a student who studied 3.5 hours
new_student = np.array([[3.5]])
prediction = model.predict(new_student)
probability = model.predict_proba(new_student)

print(f"Prediction: {'Pass' if prediction[0] == 1 else 'Fail'}")
print(f"Probability of passing: {probability[0][1]:.2%}")
# Output: Prediction: Pass, Probability: 65.32%
```

**[PLACEHOLDER FOR CLASSIFICATION VISUALIZATION]**  
*Create a 2D decision boundary plot:*
- *X-axis: Study Hours (0-10)*
- *Y-axis: Previous Score (0-100)*
- *Red dots: Failed students*
- *Green dots: Passed students*
- *Purple line: Decision boundary*
- *Shaded regions: Fail zone (red) and Pass zone (green)*
- *Title: "Binary Classification: Pass/Fail Prediction"*

---

## Regression vs Classification: Visual Comparison

| Aspect | Regression | Classification |
|--------|-----------|---------------|
| **Output Type** | Continuous numbers | Discrete categories |
| **Examples** | 23.5, 156.78, -42.1 | "Cat", "Dog", "Bird" |
| **Visualization** | Line, curve | Decision boundaries |
| **Evaluation** | MAE, RMSE, R² | Accuracy, Precision, Recall |
| **Question Answered** | "How much?" | "Which category?" |

**[PLACEHOLDER FOR SIDE-BY-SIDE COMPARISON]**  
*Create two graphs side by side:*
- *Left: Regression plot (continuous line through points)*
- *Right: Classification plot (discrete categories separated by boundary)*
- *Both use same-style axes and colors for easy comparison*
- *Label clearly: "Continuous Output" vs "Discrete Output"*

---

## Unsupervised Learning

### What is Unsupervised Learning?

**Unsupervised Learning** is like exploring without a teacher. You're given:
- **Input data (X)**: Features/attributes
- **No labels (No Y)**: No correct answers provided
- **Goal**: Find hidden patterns, structures, or relationships in data

### Real-World Analogy
Imagine sorting a mixed pile of LEGO bricks:
- No one told you how to group them
- You observe similarities (color, size, shape)
- You create your own categories
- You discover patterns on your own

### How It Works

```python
# No labels provided!
Input: Data without any categories
[Customer Age, Income, Spending] → ?
[25, $40k, $500/month] → Which group?
[45, $80k, $1200/month] → Which group?
[30, $45k, $600/month] → Which group?

Model finds: "These customers form 3 natural groups!"
- Group 1: Young, low income, low spending
- Group 2: Middle-aged, high income, high spending  
- Group 3: Mixed age, medium income, medium spending
```

**[PLACEHOLDER FOR UNSUPERVISED LEARNING DIAGRAM]**  
*Create a visualization showing:*
- *Top: Unlabeled scattered data points (all same color)*
- *Arrow pointing down to "Pattern Discovery"*
- *Bottom: Same points now colored in 3-4 clusters*
- *Title: "Unsupervised Learning: Finding Hidden Patterns"*

---

## Types of Unsupervised Learning

### 1. Clustering

**Grouping similar data points together** based on their characteristics.

#### Use Cases
- **Customer Segmentation**: Group customers by behavior for targeted marketing
- **Image Segmentation**: Separate objects in images
- **Document Organization**: Group similar articles or documents
- **Anomaly Detection**: Identify outliers (data points that don't fit any cluster)

#### Common Algorithms
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
- Mean Shift
- Gaussian Mixture Models

#### Real Example: Customer Segmentation

```python
from sklearn.cluster import KMeans
import numpy as np

# Customer data: [Age, Annual Income (k$), Spending Score (1-100)]
customers = np.array([
    [25, 40, 39],
    [45, 80, 81],
    [30, 50, 6],
    [35, 75, 77],
    [28, 42, 40],
    [50, 85, 10]
])

# Create 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(customers)

print("Customer clusters:", clusters)
# Output: [0, 1, 2, 1, 0, 2]
# Customer 0 and 4 are similar (Cluster 0)
# Customer 1 and 3 are similar (Cluster 1)
# Customer 2 and 5 are similar (Cluster 2)
```

**[PLACEHOLDER FOR CLUSTERING DIAGRAM]**  
*Create a scatter plot:*
- *X-axis: Annual Income*
- *Y-axis: Spending Score*
- *Points colored by cluster (3 different colors)*
- *Cluster centroids marked with X*
- *Title: "K-Means Clustering: Customer Segmentation"*
- *Legend showing what each cluster represents*

---

### 2. Dimensionality Reduction

**Reducing the number of features while preserving important information.**

#### Why Reduce Dimensions?
- **Visualization**: Can't plot 100-dimensional data, reduce to 2D/3D
- **Performance**: Fewer features = faster training
- **Remove Noise**: Eliminate irrelevant features
- **Storage**: Less data to store
- **Avoid Curse of Dimensionality**: Too many features can hurt performance

#### Use Cases
- **Data Visualization**: Plot high-dimensional data in 2D/3D
- **Feature Extraction**: Reduce from 1000 features to 50 essential ones
- **Compression**: Reduce image/data size
- **Preprocessing**: Prepare data for other ML algorithms

#### Common Algorithms
- **PCA (Principal Component Analysis)**: Most popular
- **t-SNE**: Great for visualization
- **UMAP**: Faster alternative to t-SNE
- **Autoencoders**: Neural network approach

#### Real Example: Image Compression

```python
from sklearn.decomposition import PCA
import numpy as np

# Image with 1000 pixels (1000 dimensions)
image_data = np.random.rand(100, 1000)  # 100 images, 1000 pixels each

# Reduce to 50 dimensions (95% variance retained)
pca = PCA(n_components=50)
reduced_data = pca.fit_transform(image_data)

print(f"Original shape: {image_data.shape}")  # (100, 1000)
print(f"Reduced shape: {reduced_data.shape}")  # (100, 50)
print(f"Variance retained: {pca.explained_variance_ratio_.sum():.2%}")
# Reduced 95% of storage while keeping 95% of information!
```

**[PLACEHOLDER FOR PCA VISUALIZATION]**  
*Create a before/after comparison:*
- *Left: 3D scatter plot of original data (complex, many dimensions implied)*
- *Right: 2D scatter plot after PCA (clearer structure visible)*
- *Arrow between them labeled "PCA Transformation"*
- *Title: "Dimensionality Reduction: 100D → 2D"*

---

### 3. Anomaly Detection

**Identifying unusual data points that don't fit normal patterns.**

#### Use Cases
- **Fraud Detection**: Unusual credit card transactions
- **Network Security**: Detecting cyber attacks
- **Manufacturing**: Identifying defective products
- **Healthcare**: Detecting abnormal patient vitals
- **System Monitoring**: Finding server failures

#### Common Algorithms
- Isolation Forest
- One-Class SVM
- Local Outlier Factor (LOF)
- Autoencoders

#### Real Example: Credit Card Fraud

```python
from sklearn.ensemble import IsolationForest
import numpy as np

# Transaction data: [Amount, Time of Day, Location Distance from Home]
transactions = np.array([
    [50, 14, 2],      # Normal
    [120, 10, 5],     # Normal
    [45, 18, 1],      # Normal
    [5000, 3, 500],   # ANOMALY! Large amount, odd time, far location
    [80, 12, 3],      # Normal
    [200, 15, 10]     # Normal
])

# Train anomaly detector
detector = IsolationForest(contamination=0.1, random_state=42)
predictions = detector.fit_predict(transactions)

# -1 = Anomaly, 1 = Normal
for i, pred in enumerate(predictions):
    status = "ANOMALY" if pred == -1 else "Normal"
    print(f"Transaction {i}: {status}")
# Output: Transaction 3: ANOMALY
```

**[PLACEHOLDER FOR ANOMALY DETECTION DIAGRAM]**  
*Create a scatter plot:*
- *Most points clustered together (blue dots)*
- *Few points far from cluster (red dots with warning icons)*
- *Circle showing "normal region"*
- *Arrows pointing to anomalies with labels*
- *Title: "Anomaly Detection: Identifying Outliers"*

---

### 4. Association Rule Learning

**Discovering interesting relationships and associations between variables.**

#### The Famous Example
**Walmart's Beer and Diapers**: Analysis found that customers who buy diapers often buy beer too. Why? Fathers sent to buy diapers would pick up beer. Result: Store placed these items near each other → increased sales!

#### Use Cases
- **Market Basket Analysis**: "Customers who bought X also bought Y"
- **Recommendation Systems**: "If you liked this, you'll like..."
- **Web Usage Mining**: Understanding user navigation patterns
- **Medical Diagnosis**: Symptoms that occur together

#### Common Algorithms
- **Apriori Algorithm**
- **FP-Growth**
- **ECLAT**

#### Key Metrics
- **Support**: How often items appear together
- **Confidence**: If A is bought, probability B is also bought
- **Lift**: How much more likely B is bought when A is bought

#### Real Example: Market Basket Analysis

```python
# Simulated transaction data
transactions = [
    ['bread', 'milk', 'butter'],
    ['bread', 'diaper', 'beer', 'eggs'],
    ['milk', 'diaper', 'beer', 'cola'],
    ['bread', 'milk', 'diaper', 'beer'],
    ['bread', 'milk', 'diaper', 'cola']
]

# Discovered rules (simplified):
# Rule 1: {diaper} → {beer}
#   Support: 60% (appears in 3/5 transactions)
#   Confidence: 75% (when diaper bought, beer bought 75% of time)
#   Lift: 1.5 (beer is 50% more likely with diaper than alone)

# Rule 2: {bread, milk} → {butter}
#   Support: 20%
#   Confidence: 50%
```

**[PLACEHOLDER FOR ASSOCIATION RULES VISUALIZATION]**  
*Create a network graph:*
- *Nodes: Different products (bread, milk, beer, diaper, etc.)*
- *Edges: Lines connecting frequently bought together items*
- *Thickness of edges: Strength of association*
- *Labels on edges: Confidence percentage*
- *Title: "Association Rules: Market Basket Analysis"*

---

## Semi-Supervised Learning

### What is Semi-Supervised Learning?

**Combination of supervised and unsupervised learning** using both labeled and unlabeled data.

### Why Use It?

**The Labeling Problem:**
- Labeling data is **expensive** and **time-consuming**
- Example: Medical images need expert doctors to label (hours per image)
- Example: Labeling 1 million images would take years!

**The Solution:**
- Use **small amount of labeled data** (expensive but accurate)
- Use **large amount of unlabeled data** (cheap and abundant)
- Model learns from both!

### Real-World Analogy
Learning a new language:
- Teacher explains 100 words (labeled data)
- You read thousands of books in that language (unlabeled data)
- You use context from books to understand new words
- Much more effective than just those 100 taught words!

### How It Works

```python
Training Data:
├── Labeled: 100 images with labels (10% - expensive to create)
└── Unlabeled: 10,000 images without labels (90% - easy to collect)

Process:
1. Train on 100 labeled images (supervised)
2. Use model to predict labels for unlabeled images
3. Add confident predictions to training set
4. Retrain with larger dataset
5. Repeat until convergence

Result: Performance much better than using only 100 labeled images!
```

### Use Cases
- **Web Content Classification**: Few labeled pages, millions of unlabeled
- **Speech Recognition**: Limited transcribed audio, unlimited raw audio
- **Medical Image Analysis**: Few expert-labeled scans, many unlabeled
- **Text Classification**: Small training set, huge corpus of text

### Cost Comparison

| Approach | Labeled Data Needed | Cost | Performance |
|----------|-------------------|------|-------------|
| **Supervised** | 10,000 | $$$$$  | Excellent |
| **Semi-Supervised** | 100 + 10,000 unlabeled | $$ | Very Good |
| **Unsupervised** | 0 | $ | Limited |

**[PLACEHOLDER FOR SEMI-SUPERVISED DIAGRAM]**  
*Create a visual comparison:*
- *Three sections: Supervised, Semi-Supervised, Unsupervised*
- *Show data as dots: Labeled (colored), Unlabeled (gray)*
- *Supervised: All colored*
- *Semi-Supervised: Few colored, many gray*
- *Unsupervised: All gray*
- *Show decision boundaries for each*
- *Add cost icons ($) and performance stars (★)*

---

## Reinforcement Learning

### What is Reinforcement Learning?

**Learning by trial and error through interaction with an environment**, receiving rewards for good actions and penalties for bad ones.

### Real-World Analogy
Training a dog:
- Dog tries different actions (sit, bark, jump)
- You give treats (reward) for good behavior
- You say "no" (penalty) for bad behavior  
- Over time, dog learns which actions get treats
- Dog learns the **best strategy** to maximize treats

### Key Components

```
Agent ←→ Environment
  ↓
State → Action → Reward → New State
  ↑                           ↓
  └───────── Learn ──────────┘
```

- **Agent**: The learner/decision maker (the dog)
- **Environment**: The world the agent interacts with
- **State**: Current situation
- **Action**: What the agent can do
- **Reward**: Feedback (positive/negative)
- **Policy**: Strategy for choosing actions
- **Goal**: Maximize cumulative reward

**[PLACEHOLDER FOR RL CYCLE DIAGRAM]**  
*Create a circular flow diagram:*
- *Center: "Agent" (brain icon)*
- *Surrounding: "Environment" (world icon)*
- *Arrows forming cycle: Agent → Action → Environment → State + Reward → Agent*
- *Show examples at each step (game character, move left, game world, score +10)*
- *Title: "Reinforcement Learning Loop"*

---

### How It Works: Example

**Teaching a robot to walk:**

```python
Initial State: Robot standing still

Iteration 1:
  Action: Move left leg forward
  Result: Robot falls
  Reward: -10 (penalty)
  Learning: "That wasn't good"

Iteration 2:
  Action: Shift weight, then move leg
  Result: One step forward
  Reward: +1
  Learning: "Better!"

Iteration 100:
  Action: Optimized walking sequence
  Result: Walks smoothly
  Reward: +100
  Learning: "Perfect!"

After 10,000 iterations:
  Robot has learned optimal walking strategy
```

### Real Applications

| Application | Agent | Actions | Reward |
|-------------|-------|---------|--------|
| **Game Playing** | AI Player | Move left, right, jump | Win game: +1000, Lose: -1000 |
| **Autonomous Driving** | Self-driving car | Accelerate, brake, steer | Stay in lane: +1, Crash: -1000 |
| **Robot Navigation** | Robot | Move forward, turn, stop | Reach goal: +100, Hit wall: -10 |
| **Stock Trading** | Trading bot | Buy, sell, hold | Profit: +$, Loss: -$ |
| **Recommendation** | Recommender | Suggest item A, B, or C | User clicks: +1, Ignores: 0 |

### Types of Reinforcement Learning

1. **Model-Free RL**
   - Agent learns without understanding the environment
   - Examples: Q-Learning, SARSA, Policy Gradient

2. **Model-Based RL**
   - Agent builds a model of the environment
   - Plans ahead using the model
   - More sample-efficient

3. **Deep Reinforcement Learning**
   - Combines RL with Deep Neural Networks
   - Can handle complex environments (images, video)
   - Examples: DQN, A3C, PPO

### Famous Achievements

🏆 **AlphaGo** (2016): Defeated world Go champion  
🎮 **DQN** (2013): Learned to play Atari games from pixels  
🤖 **OpenAI Five** (2018): Beat professional Dota 2 players  
♟️ **AlphaZero** (2017): Mastered chess, Go, and shogi from scratch  
🦾 **Robotics**: Robots learning to manipulate objects

### Code Example: Simple Q-Learning

```python
import numpy as np

# Simple grid world: Robot finding treasure
# 0 = empty, 1 = wall, 9 = treasure
grid = np.array([
    [0, 0, 0, 9],
    [0, 1, 0, 0],
    [0, 0, 0, 0]
])

# Q-table: stores value of each action in each state
Q = np.zeros((12, 4))  # 12 positions, 4 actions (up, down, left, right)

# Learning parameters
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1  # exploration rate

# Training for 1000 episodes
for episode in range(1000):
    state = 0  # start position
    while state != 3:  # until reaching treasure
        # Choose action (explore vs exploit)
        if np.random.random() < epsilon:
            action = np.random.randint(4)  # explore
        else:
            action = np.argmax(Q[state])  # exploit
        
        # Take action, get reward
        next_state, reward = take_action(state, action)
        
        # Update Q-value
        Q[state, action] = Q[state, action] + learning_rate * (
            reward + discount_factor * np.max(Q[next_state]) - Q[state, action]
        )
        
        state = next_state

# After training, robot knows optimal path to treasure!
```

**[PLACEHOLDER FOR RL GRID WORLD]**  
*Create a grid visualization:*
- *Grid with start position (green), treasure (gold), walls (black)*
- *Show learned path with arrows*
- *Heat map showing Q-values (darker = higher value)*
- *Title: "Q-Learning: Finding Optimal Path"*

---

## Comparison and When to Use Each

### Quick Decision Guide

**[PLACEHOLDER FOR DECISION FLOWCHART]**  
*Create an interactive flowchart:*
- *Start: "What data do you have?"*
- *Branch 1: "Labeled data" → "What's your output?" → "Number" = Regression, "Category" = Classification*
- *Branch 2: "No labels" → "What's your goal?" → "Group similar" = Clustering, "Reduce features" = Dimensionality Reduction, etc.*
- *Branch 3: "Some labels" → Semi-Supervised*
- *Branch 4: "Sequential decisions" → Reinforcement Learning*
- *Use colors and icons for each outcome*

### Comprehensive Comparison Table

| Type | Labeled Data? | Goal | Example Problems | Common Algorithms |
|------|--------------|------|------------------|-------------------|
| **Supervised - Regression** | ✅ Yes (numerical) | Predict continuous values | House prices, temperature, stock prices | Linear Regression, Random Forest, Neural Networks |
| **Supervised - Classification** | ✅ Yes (categorical) | Predict categories | Spam detection, image recognition, diagnosis | Logistic Regression, SVM, Decision Trees, CNN |
| **Unsupervised - Clustering** | ❌ No | Group similar items | Customer segmentation, document grouping | K-Means, DBSCAN, Hierarchical |
| **Unsupervised - Dimensionality Reduction** | ❌ No | Reduce features | Visualization, compression | PCA, t-SNE, UMAP |
| **Unsupervised - Anomaly Detection** | ❌ No | Find outliers | Fraud detection, defect detection | Isolation Forest, One-Class SVM |
| **Unsupervised - Association** | ❌ No | Find relationships | Market basket, recommendations | Apriori, FP-Growth |
| **Semi-Supervised** | 🔄 Partially | Leverage unlabeled data | Image/text classification with limited labels | Label Propagation, Self-training |
| **Reinforcement** | ⚡ Sequential | Learn optimal actions | Game playing, robotics, trading | Q-Learning, DQN, PPO |

### When to Use What?

#### Use **Supervised Learning** when:
✅ You have labeled training data  
✅ You want to predict specific outputs  
✅ Relationship between input and output exists  
✅ Problem is well-defined with clear success metrics  

#### Use **Unsupervised Learning** when:
✅ You don't have labeled data (or labeling is too expensive)  
✅ You want to explore data structure  
✅ Goal is to find patterns, not predict specific values  
✅ You need preprocessing for other ML tasks  

#### Use **Semi-Supervised Learning** when:
✅ Labeling data is expensive/time-consuming  
✅ You have small labeled dataset but large unlabeled dataset  
✅ Unlabeled data shares distribution with labeled data  
✅ You want to improve model with limited budget  

#### Use **Reinforcement Learning** when:
✅ Problem involves sequential decision-making  
✅ You can define rewards/penalties  
✅ Agent can interact with environment  
✅ Trial and error is acceptable  
✅ Goal is long-term optimization  

---

## Real-World Project Examples

### Example 1: E-commerce Recommendation System

**Multiple ML Types Combined:**

1. **Clustering (Unsupervised)**  
   → Group customers by browsing behavior
   
2. **Association Rules (Unsupervised)**  
   → Find "frequently bought together" patterns
   
3. **Classification (Supervised)**  
   → Predict if customer will click recommendation
   
4. **Regression (Supervised)**  
   → Predict rating customer will give to product
   
5. **Reinforcement Learning**  
   → Optimize recommendation sequence to maximize purchases

### Example 2: Autonomous Vehicle

**Multiple ML Types Combined:**

1. **Classification (Supervised)**  
   → Identify objects (pedestrian, car, sign)
   
2. **Regression (Supervised)**  
   → Predict trajectory of other vehicles
   
3. **Anomaly Detection (Unsupervised)**  
   → Detect unusual situations (obstacle on road)
   
4. **Reinforcement Learning**  
   → Learn optimal driving policy (when to brake, turn, accelerate)

---

## Practical Tips for Beginners

### Starting Your ML Journey

1. **Start with Supervised Learning**
   - Easiest to understand and implement
   - Clear success metrics
   - Most tutorials and resources available

2. **Get Good at Data Preprocessing**
   - 80% of ML work is data preparation
   - Learn to clean, transform, and explore data

3. **Master One Algorithm in Each Category**
   - Regression: Linear Regression
   - Classification: Logistic Regression or Decision Tree
   - Clustering: K-Means
   - Then expand to others

4. **Work on Real Projects**
   - Kaggle competitions
   - Personal projects with your own data
   - Open datasets (UCI ML Repository, Kaggle Datasets)

5. **Understand When NOT to Use ML**
   - Sometimes simple rules work better
   - Consider computational cost
   - Interpretability requirements

---

## 🧠 Quick Quiz

Test your understanding:

1. What's the main difference between supervised and unsupervised learning?
2. When would you use regression vs classification?
3. Give an example of a real-world problem for each ML type.
4. Why would you use semi-supervised learning instead of supervised learning?
5. What type of learning would you use for a robot learning to play chess?

<details>
<summary>Click for answers</summary>

1. Supervised has labeled data (input + correct output), unsupervised doesn't (input only)
2. Regression for continuous numerical output (price, temperature), Classification for categorical output (spam/not spam, dog/cat)
3. 
   - Supervised: Email spam filter
   - Unsupervised: Customer segmentation
   - Semi-supervised: Image classification with limited labels
   - Reinforcement: Game-playing AI
4. When labeling data is expensive but you have lots of unlabeled data available
5. Reinforcement Learning (learns through playing games, receiving rewards for wins)

</details>

---

## Summary

🎯 **Key Takeaways:**

- **Supervised Learning**: Has labels, predicts outputs (regression/classification)
- **Unsupervised Learning**: No labels, finds patterns (clustering, dimensionality reduction, anomaly detection, association)
- **Semi-Supervised Learning**: Mix of labeled and unlabeled data, cost-effective
- **Reinforcement Learning**: Learns through trial and error, maximizes rewards

Most real-world systems **combine multiple types** of machine learning!

---

*Previous: [← Introduction to ML](./01_introduction_to_ml.md)*  
*Next: [Applications of Machine Learning →](./03_applications_of_ml.md)*
