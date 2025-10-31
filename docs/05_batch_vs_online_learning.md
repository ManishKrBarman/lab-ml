# Chapter 5: Training Strategies - Batch vs Online Learning

## 📖 Table of Contents
- [Overview](#overview)
- [Batch Learning](#batch-learning)
- [Online Learning](#online-learning)
- [Mini-Batch Learning](#mini-batch-learning)
- [Comparison and Trade-offs](#comparison-and-trade-offs)
- [When to Use Each](#when-to-use-each)

---

## Overview

Another fundamental way to categorize machine learning systems is by **how they learn from data over time**. This is especially important for production systems that need to adapt to changing conditions.

### The Three Approaches

```
Training Strategies
    ├── Batch Learning (Offline Learning)
    │   └── Train on entire dataset at once
    │
    ├── Online Learning (Incremental Learning)
    │   └── Train on data points one at a time
    │
    └── Mini-Batch Learning
        └── Train on small batches of data
```

**[PLACEHOLDER FOR TRAINING STRATEGIES OVERVIEW]**  
*Create a visual comparison:*
- *Three panels showing how data flows in each approach*
- *Batch: All data → Model*
- *Online: Data stream → Model (updating continuously)*
- *Mini-Batch: Small chunks → Model*
- *Use arrows and animations to show flow*

---

## Batch Learning

### What is Batch Learning?

**Batch Learning** (also called **offline learning**) is the traditional approach where the model is trained on the **entire dataset at once**. Once trained, the model is deployed and doesn't learn anymore - it just makes predictions.

### Real-World Analogy

**College Exam Preparation**:
- You study all semester material over several weeks
- Take the final exam
- After exam, you don't update your knowledge
- If curriculum changes, you must study everything again from scratch

### How It Works

```
Step 1: Collect Data
├── Gather all historical data
├── Clean and preprocess
└── Create training dataset

Step 2: Train Model (Offline)
├── Use powerful computers
├── Train for hours/days
├── Tune hyperparameters
└── Validate performance

Step 3: Deploy Model (Production)
├── Model is frozen
├── Makes predictions only
└── Doesn't learn from new data

Step 4: Retrain (Periodically)
├── Collect new data
├── Add to old data
├── Train new model from scratch
└── Replace old model
```

**[PLACEHOLDER FOR BATCH LEARNING WORKFLOW]**  
*Create a circular workflow diagram:*
- *Collect Data → Train Model → Deploy → Monitor Performance → Collect More Data → Retrain*
- *Show "Development" and "Production" zones*
- *Highlight that learning only happens in development*

---

### Characteristics of Batch Learning

#### ✅ Advantages

1. **Simpler to Implement**
   - Standard approach, lots of tools and resources
   - Easier to debug and validate

2. **Better for Complex Models**
   - Can use the full dataset for training
   - Better at finding global patterns
   - Can train for many epochs

3. **Stable Predictions**
   - Model doesn't change in production
   - Consistent behavior
   - Easier to test and validate

4. **Resource Efficient During Inference**
   - No training happening in production
   - Predictable resource usage
   - Can use less powerful production servers

5. **Better Optimization**
   - Can use the entire dataset for each update
   - More stable convergence
   - Can use complex optimization techniques

#### ❌ Disadvantages

1. **Doesn't Adapt to New Patterns**
   - Can't learn from new data in real-time
   - Model gets stale over time
   - Miss emerging trends

2. **Resource Intensive Training**
   - Needs all data loaded in memory (or accessible)
   - Can take hours or days to train
   - Requires powerful hardware

3. **Delayed Updates**
   - Must wait for retraining cycle
   - Can't respond to sudden changes
   - Retraining can be expensive

4. **Scalability Challenges**
   - As dataset grows, retraining becomes slower
   - May need to sample or discard old data
   - Training time grows with data size

5. **Cold Start Problem**
   - Can't make good predictions until fully trained
   - Need substantial historical data

---

### When Batch Learning is Used

#### Perfect For:

1. **Stable Environments**
   - Data distribution doesn't change rapidly
   - Example: Historical sales forecasting

2. **Offline Processing**
   - No need for real-time adaptation
   - Example: Monthly report generation

3. **Small to Medium Datasets**
   - Can train quickly on full dataset
   - Example: Scientific research projects

4. **When Accuracy is Critical**
   - Need time to validate model thoroughly
   - Example: Medical diagnosis systems

#### Real-World Examples:

| Application | Why Batch Learning | Retraining Frequency |
|-------------|-------------------|---------------------|
| **Spam Filters** | Patterns change slowly | Weekly/Monthly |
| **Credit Scoring** | Historical patterns stable | Quarterly |
| **Image Classification** | Objects don't change | When new categories added |
| **Recommendation Systems** | Can retrain nightly | Daily |
| **Weather Prediction** | Seasonal patterns stable | When model architecture changes |

---

### The Model Staleness Problem

**Problem**: Models become outdated over time

```python
January 2024:
Model trained on 2023 data
Accuracy: 95%

June 2024:
Same model, new data patterns emerged
Accuracy: 87% (dropped!)

Solution: Retrain
Retrain on Jan-May 2024 data
Accuracy: 94% (recovered!)
```

**[PLACEHOLDER FOR MODEL STALENESS GRAPH]**  
*Create a line graph:*
- *X-axis: Time (months)*
- *Y-axis: Model Accuracy*
- *Line shows gradual decline over time*
- *Vertical bars showing retraining points (accuracy jumps back up)*
- *Label periods of "Model Rot" or "Model Drift"*

---

### Practical Implementation

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Step 1: Load entire dataset
data = load_all_data()  # Could be millions of rows
X_train, X_test, y_train, y_test = train_test_split(data)

# Step 2: Train model on entire training set
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)  # Could take hours!

# Step 3: Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2%}")

# Step 4: Save model for deployment
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Step 5: Deploy (model is now frozen)
# In production, model only predicts, never learns

# Step 6: Retrain later (manual process)
# Must schedule retraining, redeploy
```

---

## Online Learning

### What is Online Learning?

**Online Learning** (also called **incremental learning**) trains the model **incrementally** on new data as it arrives, either one sample at a time or in small groups. The model continuously adapts.

### Real-World Analogy

**Learning to Drive**:
- You don't wait to experience all possible scenarios before driving
- You learn incrementally: each drive teaches you something new
- You adapt to new road conditions, traffic patterns
- Your driving improves continuously with experience

### How It Works

```
Continuous Loop:
    New Data Arrives
         ↓
    Update Model (incremental)
         ↓
    Make Predictions
         ↓
    Observe Outcome
         ↓
    Learn from Feedback
         ↓
    (repeat)
```

**[PLACEHOLDER FOR ONLINE LEARNING FLOW]**  
*Create an animated circular diagram:*
- *Data stream flowing continuously*
- *Model in center updating in real-time*
- *Arrows showing continuous feedback loop*
- *Highlight "always learning" nature*

---

### Characteristics of Online Learning

#### ✅ Advantages

1. **Adapts to Change**
   - Responds to new patterns immediately
   - Handles concept drift naturally
   - Stays current with latest trends

2. **Memory Efficient**
   - Don't need to store all historical data
   - Process data, update model, discard data
   - Can work with data streams

3. **Scalable**
   - Can handle massive datasets (don't need all in memory)
   - Training time doesn't grow with dataset size
   - Out-of-core learning possible

4. **Fast Deployment**
   - Can start with simple model, improve over time
   - No need to wait for full training
   - Continuous improvement

5. **Real-Time Learning**
   - Learns from user interactions immediately
   - Personalization happens in real-time

#### ❌ Disadvantages

1. **Sensitive to Data Quality**
   - Bad data immediately affects model
   - Outliers can skew learning
   - Can be "poisoned" by malicious data

2. **Harder to Optimize**
   - Can't use full dataset for each update
   - May get stuck in local minima
   - Requires careful learning rate tuning

3. **More Complex to Implement**
   - Need streaming infrastructure
   - Harder to debug
   - Version control challenging

4. **Resource Intensive in Production**
   - Continuous training requires compute
   - Need to monitor model health
   - Can be expensive at scale

5. **Harder to Validate**
   - Model constantly changing
   - A/B testing more complex
   - Reproducibility challenges

---

### Key Concept: Learning Rate

**Learning Rate** controls how much the model changes with each update.

```python
# High Learning Rate (0.5)
- Large updates from each sample
- Adapts quickly
- But unstable, can forget old knowledge
- Risk: Model can be misled by outliers

# Low Learning Rate (0.01)
- Small updates from each sample
- Stable, retains old knowledge
- But adapts slowly to changes
- Risk: Slow to catch new trends

# Optimal: Decreasing learning rate
- Start high: learn quickly
- Decrease over time: stabilize
```

**[PLACEHOLDER FOR LEARNING RATE VISUALIZATION]**  
*Create three graphs side-by-side:*
- *Low LR: Smooth convergence, slow adaptation*
- *Optimal LR: Good balance*
- *High LR: Fast but unstable, oscillates*
- *Show model updates as the data stream arrives*

---

### Concept Drift

**Concept Drift**: When the statistical properties of the target variable change over time.

#### Types of Drift:

1. **Sudden Drift**
   ```
   Pattern A ────┐
                 └──── Pattern B ───────
   ```
   Example: COVID-19 sudden impact on shopping patterns

2. **Gradual Drift**
   ```
   Pattern A ─────╱
                 ╱
                ╱── Pattern B
   ```
   Example: Fashion trends evolving slowly

3. **Incremental Drift**
   ```
   Pattern A ──╱─╲─╱─╲─╱─╲─╱── Pattern B
   ```
   Example: Seasonal patterns with random variations

4. **Recurring Patterns**
   ```
   Pattern A ─╲  ╱─╲  ╱─╲
              ╱╲╱   ╲╱   ╲╱
   Pattern B ─  ─────  ─────
   ```
   Example: Weekly shopping patterns (weekday vs weekend)

**[PLACEHOLDER FOR CONCEPT DRIFT TYPES]**  
*Create 4 line graphs showing each drift type:*
- *X-axis: Time*
- *Y-axis: Pattern/Distribution*
- *Clearly show how patterns change for each type*
- *Use color coding to distinguish old vs new patterns*

---

### Out-of-Core Learning

**Out-of-Core Learning**: Training on datasets too large to fit in memory.

```python
# Dataset: 100 GB (can't fit in 16 GB RAM)

# Traditional Batch: ❌ Out of memory error

# Online Learning: ✅ Process in chunks
from sklearn.linear_model import SGDClassifier

model = SGDClassifier()

# Read data in chunks
for chunk in read_data_in_chunks(chunk_size=10000):
    X, y = chunk
    model.partial_fit(X, y, classes=[0, 1])
    # Process chunk, update model, chunk is garbage collected

# Can handle unlimited dataset size!
```

---

### Practical Implementation

```python
from sklearn.linear_model import SGDClassifier
import numpy as np

# Initialize model
model = SGDClassifier(loss='log', learning_rate='optimal')

# Simulate data stream
for i in range(1000):  # 1000 data points arriving over time
    # New data arrives
    X_new = np.array([[...]])  # 1 sample
    y_new = np.array([...])
    
    # Update model incrementally
    model.partial_fit(X_new, y_new, classes=[0, 1])
    
    # Make predictions with updated model
    prediction = model.predict(X_new)
    
    # Model has learned from this example
    # and is ready for the next one!

# Model continuously adapts to new patterns
```

---

### Real-World Examples

| Application | Why Online Learning | Update Frequency |
|-------------|-------------------|------------------|
| **Stock Trading** | Prices change by second | Real-time |
| **News Recommendation** | Trending topics emerge | Minutes |
| **Fraud Detection** | New fraud patterns | Real-time |
| **Search Engines** | User preferences evolve | Continuous |
| **Social Media Feeds** | User interests change | Real-time |
| **IoT Sensors** | Massive continuous data | Continuous |
| **Chatbots** | Learn from conversations | After each chat |

---

## Mini-Batch Learning

### What is Mini-Batch Learning?

**Best of both worlds**: Train on small batches of data, combining benefits of batch and online learning.

```
Instead of:
- Batch: All 1 million samples at once
- Online: 1 sample at a time

Mini-Batch: Groups of 32, 64, 128, or 256 samples
```

### How It Works

```python
# Dataset: 10,000 samples
# Mini-batch size: 100

For epoch in range(num_epochs):
    Shuffle data
    For each mini-batch of 100 samples:
        1. Forward pass (compute predictions)
        2. Compute loss
        3. Backward pass (compute gradients)
        4. Update model parameters
```

### Why Mini-Batches?

#### Better than Single Samples (Online):
- ✅ More stable gradient estimates
- ✅ Better use of parallel processing (GPUs)
- ✅ Faster convergence
- ✅ Less noisy updates

#### Better than Full Batch:
- ✅ Faster iterations
- ✅ Can handle larger datasets
- ✅ Better generalization (slight noise helps)
- ✅ Memory efficient

**[PLACEHOLDER FOR MINI-BATCH VISUALIZATION]**  
*Create a comparison showing gradient updates:*
- *Panel 1: Single sample (very noisy path to optimum)*
- *Panel 2: Mini-batch (smoother path, good balance)*
- *Panel 3: Full batch (smoothest but slowest)*
- *Show convergence paths on loss landscape*

---

### Choosing Batch Size

| Batch Size | Pros | Cons | Use Case |
|-----------|------|------|----------|
| **1** (Online) | Frequent updates, adapts fast | Very noisy, unstable | Streaming data |
| **32** | Good balance, fast | Some noise | General purpose |
| **64-128** | Stable, GPU efficient | Medium speed | Deep learning (small models) |
| **256-512** | Very stable, good for GPUs | Slower per epoch | Deep learning (large models) |
| **Full Batch** | Most stable | Slow, memory intensive | Small datasets |

### Power of 2 Rule
Use batch sizes that are powers of 2 (32, 64, 128, 256) for better GPU utilization!

---

### Practical Implementation

```python
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load data
X_train, y_train = load_data()

# Create data pipeline with mini-batches
BATCH_SIZE = 32

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(BATCH_SIZE)

# Model trains on mini-batches automatically
model = create_model()
model.fit(dataset, epochs=10)

# Behind the scenes:
# - Takes 32 samples at a time
# - Computes gradients
# - Updates model
# - Repeats for all batches
# - Continues for 10 epochs
```

---

## Comparison and Trade-offs

### Comprehensive Comparison

| Aspect | Batch Learning | Online Learning | Mini-Batch Learning |
|--------|---------------|----------------|-------------------|
| **Data Processing** | All at once | One at a time | Small groups |
| **Memory Required** | High | Low | Medium |
| **Training Speed** | Slow (large batches) | Fast (per sample) | Balanced |
| **Convergence** | Stable | Unstable | Stable |
| **Adaptation** | None (frozen) | Immediate | Quick |
| **Concept Drift** | Struggles | Handles well | Handles well |
| **Implementation** | Simple | Complex | Moderate |
| **GPU Efficiency** | Good | Poor | Excellent |
| **Retraining** | Manual, expensive | Automatic | Automatic |
| **Large Datasets** | Challenging | Excellent | Good |
| **Production Cost** | Low (inference only) | High (continuous training) | Medium |

---

### Visual Comparison: Update Patterns

**[PLACEHOLDER FOR UPDATE PATTERNS COMPARISON]**  
*Create three timelines showing model updates:*

*Timeline 1: Batch Learning*
- *Long training period (dark)*
- *Deployment period (light, flat line - no learning)*
- *Another long training period*
- *Show gaps between retraining*

*Timeline 2: Online Learning*
- *Continuous small updates (many tiny increments)*
- *Show constant learning curve*
- *No distinction between training/deployment*

*Timeline 3: Mini-Batch*
- *Regular medium-sized updates*
- *Balance between batch and online*

---

### The Trade-off Triangle

**[PLACEHOLDER FOR TRADE-OFF TRIANGLE]**  
*Create a triangle with three corners:*
- *Corner 1: "Training Speed" (how fast to train)*
- *Corner 2: "Stability" (how stable the updates)*
- *Corner 3: "Adaptability" (how quickly adapts to change)*
- *Plot Batch, Online, and Mini-Batch in the triangle*
- *Show no single method dominates all three*

---

## When to Use Each

### Decision Framework

```
START HERE
    |
    ↓
Need real-time adaptation?
    ├── Yes ─→ Data is continuous stream?
    │           ├── Yes → Online Learning
    │           └── No → Mini-Batch Learning
    │
    └── No ─→ Have large dataset?
                ├── Yes → Can fit in memory?
                │         ├── Yes → Batch or Mini-Batch
                │         └── No → Online (out-of-core)
                └── No → Batch Learning
```

**[PLACEHOLDER FOR DECISION FLOWCHART]**  
*Create an interactive decision tree with the above logic*

---

### Use Cases by Industry

#### Batch Learning Best For:
- 📊 **Business Intelligence**: Monthly/quarterly reports
- 🏥 **Healthcare**: Disease prediction models (patterns stable)
- 🏦 **Credit Scoring**: Models updated periodically
- 🔬 **Scientific Research**: Historical data analysis
- 📸 **Image Classification**: Object categories don't change

#### Online Learning Best For:
- 💰 **Stock Trading**: Real-time price predictions
- 🛡️ **Cybersecurity**: Detect new attack patterns
- 📰 **News Feed**: Trending topics
- 🎮 **Game AI**: Adapt to player behavior
- 📱 **Mobile Apps**: Personalization
- 🌐 **IoT**: Sensor data streams

#### Mini-Batch Best For:
- 🤖 **Deep Learning**: All neural networks
- 🗣️ **Speech Recognition**: Large audio datasets
- 🌐 **Web Search**: Large-scale ranking
- 🎭 **Recommendation Systems**: Updated regularly
- 🚗 **Autonomous Vehicles**: Continuous improvement

---

## Hybrid Approaches

Real production systems often combine approaches:

### Example: Netflix Recommendation System

```
Layer 1: Batch Learning (Nightly)
├── Train on all historical data
├── Build user/movie embeddings
└── Deploy updated model

Layer 2: Online Learning (Real-time)
├── User watches a movie
├── Update user preferences instantly
└── Personalize recommendations

Result: Stable + Adaptive
```

### Example: Fraud Detection

```
Base Model: Batch (Weekly)
├── Train on historical fraud patterns
└── Catches known fraud types

Online Layer: Real-time
├── Learn from new fraud attempts
└── Catch emerging patterns

Fallback: Human review for uncertain cases
```

---

## Monitoring and Model Management

### For Batch Models

```python
# Monitor model performance
def monitor_batch_model():
    metrics = {
        'accuracy': [],
        'dates': []
    }
    
    # Check weekly
    for week in range(52):
        current_accuracy = evaluate_model()
        metrics['accuracy'].append(current_accuracy)
        
        # Alert if accuracy drops
        if current_accuracy < threshold:
            send_alert("Model performance degraded!")
            trigger_retraining()
```

### For Online Models

```python
# Monitor online learning
def monitor_online_model():
    # Track learning rate
    # Detect data poisoning
    # Monitor convergence
    
    if detect_anomalous_updates():
        rollback_to_previous_version()
        
    if performance_degrades():
        adjust_learning_rate()
```

---

## Practical Tips

### For Batch Learning:
1. ✅ Schedule regular retraining (don't wait for failures)
2. ✅ Keep multiple model versions for rollback
3. ✅ Monitor data distribution shifts
4. ✅ Use validation set from recent data
5. ✅ Consider warm-starting with previous model

### For Online Learning:
1. ✅ Start with lower learning rate
2. ✅ Implement safeguards against bad data
3. ✅ Use exponential moving average for stability
4. ✅ Keep checkpoints for rollback
5. ✅ Monitor learning in real-time
6. ✅ Test updates before full deployment

### For Mini-Batch Learning:
1. ✅ Experiment with batch sizes (powers of 2)
2. ✅ Shuffle data between epochs
3. ✅ Use learning rate schedules
4. ✅ Monitor GPU utilization
5. ✅ Consider data augmentation

---

## 🧠 Quick Quiz

1. What's the main difference between batch and online learning?
2. Why is online learning better for concept drift?
3. What problem does mini-batch learning solve?
4. When would you choose batch over online learning?
5. What is the learning rate and why is it important?

<details>
<summary>Click for answers</summary>

1. Batch trains on all data at once then deploys frozen model. Online trains continuously on new data as it arrives.
2. Online learning adapts immediately to new patterns, while batch models become stale until retrained.
3. Mini-batch balances stability (better than single samples) with efficiency (faster than full batch), and leverages GPUs well.
4. When data patterns are stable, you need thoroughly validated model, or when real-time adaptation isn't necessary.
5. Learning rate controls how much the model updates from each sample. Too high = unstable, too low = slow adaptation.

</details>

---

## Summary

🎯 **Key Takeaways**:

- **Batch Learning**: Train once, deploy frozen model
  - Pro: Stable, simple
  - Con: Becomes stale, can't adapt

- **Online Learning**: Train continuously on new data
  - Pro: Adapts to change, scalable
  - Con: Complex, sensitive to bad data

- **Mini-Batch Learning**: Best of both worlds
  - Pro: Stable + efficient + adapts
  - Con: Requires careful tuning

**Most Modern Systems**: Combine approaches for optimal results!

---

*Previous: [← Learning Approaches](./04_learning_approaches.md)*  
*Next: [Challenges in ML →](./06_challenges_in_ml.md)*
