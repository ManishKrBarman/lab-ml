# Chapter 9: Understanding Tensors

## 📖 Table of Contents
- [Introduction](#introduction)
- [What is a Tensor?](#what-is-a-tensor)
- [Tensor Dimensions](#tensor-dimensions)
- [Tensor Properties](#tensor-properties)
- [Tensors in Deep Learning](#tensors-in-deep-learning)
- [Working with Tensors](#working-with-tensors)
- [Real-World Applications](#real-world-applications)

---

## Introduction

**Tensors** are the fundamental data structure in modern machine learning, especially deep learning. Understanding tensors is crucial for working with frameworks like TensorFlow, PyTorch, and JAX.

> "Tensors are to Deep Learning what arrays are to programming - the basic building blocks."

### Why Tensors Matter

```
Traditional ML: Works with tables (2D data)
Deep Learning: Works with tensors (multi-dimensional data)

Examples:
- Images: 3D tensors (height × width × color channels)
- Videos: 4D tensors (time × height × width × channels)
- Text: 2D tensors (sentences × words)
- Batch of images: 4D tensors (batch × height × width × channels)
```

---

## What is a Tensor?

### Simple Definition

**A tensor is a container for numbers** - that's it! It's a generalization of scalars, vectors, and matrices to higher dimensions.

Think of tensors as nested arrays:
- Scalar (0D tensor): Just a number
- Vector (1D tensor): List of numbers
- Matrix (2D tensor): Table of numbers
- 3D+ tensor: Multi-dimensional array of numbers

### Mathematical Definition

A tensor is a geometric object that describes linear relations between geometric vectors, scalars, and other tensors. In machine learning, we use a simpler computational definition: **multi-dimensional arrays**.

**[PLACEHOLDER FOR TENSOR VISUALIZATION]**  
*Create a visual showing:*
- *0D: Single dot (scalar)*
- *1D: Line of dots (vector)*
- *2D: Grid of dots (matrix)*
- *3D: Cube of dots (3D tensor)*
- *4D: Multiple cubes (4D tensor - show conceptually)*

---

## Tensor Dimensions

### 0D Tensor (Scalar)

**Definition**: A single number

```python
import numpy as np
import torch
import tensorflow as tf

# NumPy
scalar_np = np.array(42)
print(scalar_np)        # 42
print(scalar_np.ndim)   # 0 (zero dimensions)
print(scalar_np.shape)  # () (empty tuple)

# PyTorch
scalar_torch = torch.tensor(42)
print(scalar_torch)     # tensor(42)

# TensorFlow
scalar_tf = tf.constant(42)
print(scalar_tf)        # tf.Tensor(42, shape=(), dtype=int32)
```

**Use Cases**:
- Loss value: 0.5423
- Accuracy: 0.95
- Single prediction: 42.7
- Learning rate: 0.001

**Real-World Example**:
```python
# Model training
loss = 2.456  # Scalar
epoch = 10    # Scalar
accuracy = 0.923  # Scalar
```

---

### 1D Tensor (Vector)

**Definition**: An array of numbers (one axis)

```python
# NumPy
vector_np = np.array([1, 2, 3, 4, 5])
print(vector_np)        # [1 2 3 4 5]
print(vector_np.ndim)   # 1
print(vector_np.shape)  # (5,)

# PyTorch
vector_torch = torch.tensor([1, 2, 3, 4, 5])

# TensorFlow
vector_tf = tf.constant([1, 2, 3, 4, 5])
```

**Use Cases**:
- Features of one sample: [25, 50000, 3, 1200]  # age, salary, experience, score
- Word embedding: 300-dimensional vector
- Time series: [23.5, 24.1, 23.8, 24.5, ...]  # temperatures
- Probabilities: [0.1, 0.3, 0.4, 0.2]  # class probabilities

**Visual Representation**:
```
[1, 2, 3, 4, 5]
 ↑  ↑  ↑  ↑  ↑
 Each element accessed by single index: vector[0], vector[1], ...
```

**Real-World Example**:
```python
# Single customer features
customer = np.array([35, 75000, 5, 720])  # age, income, years, credit_score

# Word embedding for "king"
king_embedding = np.random.rand(300)  # 300-dimensional vector
```

**[PLACEHOLDER FOR 1D TENSOR VISUAL]**  
*Show a horizontal array of boxes with numbers:*
- *Labeled indices: [0], [1], [2], [3], [4]*
- *Arrow showing single axis*

---

### 2D Tensor (Matrix)

**Definition**: An array of vectors (two axes: rows and columns)

```python
# NumPy
matrix_np = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(matrix_np)
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]
print(matrix_np.ndim)   # 2
print(matrix_np.shape)  # (3, 3)

# Access elements
print(matrix_np[0, 0])  # 1 (first row, first column)
print(matrix_np[1, 2])  # 6 (second row, third column)
```

**Use Cases**:
- Dataset: rows = samples, columns = features
- Image (grayscale): height × width
- Spreadsheet data
- Confusion matrix
- Weight matrix in neural network layer

**Visual Representation**:
```
     Column 0  Column 1  Column 2
Row 0    1         2         3
Row 1    4         5         6
Row 2    7         8         9

Access: matrix[row, column]
```

**Real-World Examples**:

**Example 1: Dataset**
```python
# Customer data
customers = np.array([
    [25, 50000, 2, 680],  # Customer 0
    [30, 65000, 5, 720],  # Customer 1
    [45, 85000, 10, 780]  # Customer 2
])
# Shape: (3, 4) - 3 customers, 4 features each
```

**Example 2: Grayscale Image**
```python
# 5x5 pixel grayscale image (0-255 values)
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 255, 255, 255, 0],
    [0, 255, 0, 255, 0],
    [0, 255, 255, 255, 0],
    [0, 0, 0, 0, 0]
])
# Shape: (5, 5) - 5 pixels high, 5 pixels wide
```

**[PLACEHOLDER FOR 2D TENSOR VISUAL]**  
*Show a grid/table:*
- *Rows and columns clearly labeled*
- *Sample values filled in*
- *Arrows showing two axes (rows and columns)*

---

### 3D Tensor

**Definition**: An array of matrices (three axes)

```python
# NumPy
tensor_3d = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
    [[9, 10], [11, 12]]
])
print(tensor_3d.shape)  # (3, 2, 2)
# 3 matrices, each 2x2

# Access elements
print(tensor_3d[0, 0, 0])  # 1 (first matrix, first row, first column)
print(tensor_3d[2, 1, 1])  # 12 (third matrix, second row, second column)
```

**Use Cases**:
- **Color Image**: (height, width, channels)
  - Example: (224, 224, 3) - 224×224 pixels, 3 color channels (RGB)
- **Time Series Dataset**: (samples, timesteps, features)
  - Example: (1000, 50, 5) - 1000 sequences, 50 timesteps each, 5 features
- **Multiple Grayscale Images**: (images, height, width)
  - Example: (10, 28, 28) - 10 images of 28×28 pixels

**Visual Representation**:
```
Think of a 3D tensor as:
- A stack of matrices (depth × rows × columns)
- Or a cube of numbers

Shape (3, 2, 2):
     Matrix 0    Matrix 1    Matrix 2
     [1, 2]      [5, 6]      [9, 10]
     [3, 4]      [7, 8]      [11, 12]
```

**Real-World Examples**:

**Example 1: RGB Image**
```python
# Color image: 100 pixels high, 100 pixels wide, 3 color channels
image = np.random.randint(0, 256, size=(100, 100, 3))
print(image.shape)  # (100, 100, 3)

# Access red channel of pixel (50, 50)
red_value = image[50, 50, 0]   # Red channel
green_value = image[50, 50, 1] # Green channel
blue_value = image[50, 50, 2]  # Blue channel
```

**Example 2: Video Frame**
```python
# 10 consecutive frames, each 64×64 pixels, grayscale
video_segment = np.zeros((10, 64, 64))
print(video_segment.shape)  # (10, 64, 64)

# Access pixel (32, 32) in frame 5
pixel_value = video_segment[5, 32, 32]
```

**[PLACEHOLDER FOR 3D TENSOR VISUAL]**  
*Show:*
- *Multiple stacked 2D grids (representing depth)*
- *Or a cube divided into smaller cubes*
- *Label three axes: depth, height, width*
- *Show how an RGB image maps to 3D tensor*

---

### 4D Tensor

**Definition**: An array of 3D tensors (four axes)

```python
# NumPy
tensor_4d = np.zeros((32, 224, 224, 3))
print(tensor_4d.shape)  # (32, 224, 224, 3)
```

**Use Cases**:
- **Batch of Images**: (batch_size, height, width, channels)
  - Example: (32, 224, 224, 3) - 32 images, each 224×224 pixels, 3 channels
- **Video**: (frames, height, width, channels)
  - Example: (300, 1920, 1080, 3) - 300 frames of 1920×1080 RGB video

**Visual Representation**:
```
Think of a 4D tensor as:
- Multiple 3D tensors stacked together
- A batch of images
- A collection of cubes

Shape (2, 3, 2, 2): 
2 batches, each containing a 3×2×2 tensor
```

**Real-World Example**:

**Batch of Images for Training**
```python
# 32 images in a batch, each 224×224 pixels, RGB
batch = np.random.rand(32, 224, 224, 3)
print(batch.shape)  # (32, 224, 224, 3)

# Access first image in batch
first_image = batch[0]  # Shape: (224, 224, 3)

# Access pixel (100, 100) of first image, red channel
pixel_value = batch[0, 100, 100, 0]
```

**Common Shapes**:
```python
# Different image batch formats

# TensorFlow/Keras format: (batch, height, width, channels)
tf_batch = np.zeros((32, 224, 224, 3))

# PyTorch format: (batch, channels, height, width)
torch_batch = np.zeros((32, 3, 224, 224))
```

**[PLACEHOLDER FOR 4D TENSOR VISUAL]**  
*Show:*
- *Multiple 3D cubes arranged in a row (batch dimension)*
- *Label: "Batch of Images"*
- *Highlight one cube as single image*
- *Show axes: batch, height, width, channels*

---

### 5D Tensor

**Definition**: An array of 4D tensors (five axes)

```python
# NumPy
tensor_5d = np.zeros((10, 30, 224, 224, 3))
print(tensor_5d.shape)  # (10, 30, 224, 224, 3)
```

**Use Cases**:
- **Batch of Videos**: (batch_size, frames, height, width, channels)
  - Example: (10, 30, 224, 224, 3) - 10 videos, each 30 frames, 224×224 pixels, RGB
- **Medical Imaging**: (batch, depth_slices, height, width, channels)
  - Example: MRI scans

**Real-World Example**:

**Video Dataset**
```python
# 10 video clips, each 30 frames, 128×128 pixels, RGB
videos = np.random.rand(10, 30, 128, 128, 3)
print(videos.shape)  # (10, 30, 128, 128, 3)

# Access first video
first_video = videos[0]  # Shape: (30, 128, 128, 3)

# Access first frame of first video
first_frame = videos[0, 0]  # Shape: (128, 128, 3)
```

**[PLACEHOLDER FOR 5D TENSOR CONCEPT]**  
*Show conceptual diagram:*
- *Multiple 4D structures (hard to visualize)*
- *Use nested boxes or flowchart*
- *Label: "Batch of Videos" or "Video Dataset"*

---

## Tensor Properties

### Three Key Properties: Rank, Axes, and Shape

```python
import numpy as np

tensor = np.array([
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]]
])

# 1. Rank (Number of axes/dimensions)
print(tensor.ndim)      # 3 (3D tensor)

# 2. Shape (Size along each axis)
print(tensor.shape)     # (2, 3, 2)

# 3. Data type
print(tensor.dtype)     # int64 (or int32 depending on system)
```

---

### Rank (Number of Dimensions)

**Rank = Number of axes**

```python
# Different ranks
scalar = np.array(5)            # Rank 0
vector = np.array([1, 2, 3])    # Rank 1
matrix = np.array([[1, 2]])     # Rank 2
tensor_3d = np.zeros((2, 3, 4)) # Rank 3

print(f"Scalar rank: {scalar.ndim}")      # 0
print(f"Vector rank: {vector.ndim}")      # 1
print(f"Matrix rank: {matrix.ndim}")      # 2
print(f"3D Tensor rank: {tensor_3d.ndim}") # 3
```

---

### Axes (Dimensions)

**Each axis is a dimension along which the tensor can vary**

```python
# 3D tensor: Shape (2, 3, 4)
tensor = np.zeros((2, 3, 4))

# Axis 0: 2 elements (depth)
# Axis 1: 3 elements (rows)
# Axis 2: 4 elements (columns)
```

**Understanding Axes**:
```
For shape (2, 3, 4):
- Axis 0 has length 2
- Axis 1 has length 3
- Axis 2 has length 4

Total elements: 2 × 3 × 4 = 24
```

---

### Shape

**Shape = Tuple indicating size along each axis**

```python
# Examples
scalar = np.array(42)
print(scalar.shape)  # ()

vector = np.array([1, 2, 3, 4, 5])
print(vector.shape)  # (5,)

matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix.shape)  # (2, 3)

image = np.zeros((224, 224, 3))
print(image.shape)  # (224, 224, 3)

batch = np.zeros((32, 224, 224, 3))
print(batch.shape)  # (32, 224, 224, 3)
```

---

### Data Type (dtype)

```python
# Different data types
int_tensor = np.array([1, 2, 3], dtype=np.int32)
float_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
bool_tensor = np.array([True, False, True], dtype=np.bool_)

print(int_tensor.dtype)    # int32
print(float_tensor.dtype)  # float32
print(bool_tensor.dtype)   # bool
```

**Common Types**:
- `int8`, `int16`, `int32`, `int64`: Integers
- `float16`, `float32`, `float64`: Floating-point
- `bool`: Boolean
- `uint8`: Unsigned integer (0-255, common for images)

---

## Tensors in Deep Learning

### Why Deep Learning Uses Tensors

```
Traditional ML: Tabular data (2D)
  ├─ Can't handle images well
  ├─ Can't handle sequences well
  └─ Limited to flat features

Deep Learning: Multi-dimensional tensors
  ├─ Natural representation for images (3D/4D)
  ├─ Natural representation for sequences (2D/3D)
  ├─ Can process batches efficiently
  └─ GPUs optimized for tensor operations
```

---

### Data Flow in Neural Networks

**All data is converted to tensors:**

```python
# Input → Tensor → Neural Network → Output Tensor

# Example: Image classification
import torch

# 1. Load image (becomes tensor)
image = load_image('cat.jpg')  # Shape: (3, 224, 224)

# 2. Add batch dimension
image_batch = image.unsqueeze(0)  # Shape: (1, 3, 224, 224)

# 3. Pass through network
model = torchvision.models.resnet50(pretrained=True)
output = model(image_batch)  # Shape: (1, 1000)

# 4. Get prediction
probabilities = torch.softmax(output, dim=1)
predicted_class = torch.argmax(probabilities)
```

**[PLACEHOLDER FOR NN DATA FLOW]**  
*Create a flowchart:*
- *Input image → Convert to tensor → Conv layers (showing tensor shapes)*
- *→ Pooling (shape changes) → Flatten → Dense layers*
- *→ Output tensor → Probabilities*
- *Label tensor shapes at each step*

---

### Common Tensor Operations

#### Reshaping

```python
import numpy as np

# Original tensor
tensor = np.array([1, 2, 3, 4, 5, 6])
print(tensor.shape)  # (6,)

# Reshape to 2D
tensor_2d = tensor.reshape(2, 3)
print(tensor_2d)
# [[1 2 3]
#  [4 5 6]]
print(tensor_2d.shape)  # (2, 3)

# Reshape to 3D
tensor_3d = tensor.reshape(2, 3, 1)
print(tensor_3d.shape)  # (2, 3, 1)

# Flatten
flattened = tensor_3d.flatten()
print(flattened.shape)  # (6,)
```

---

#### Slicing

```python
# 3D tensor
tensor = np.random.rand(5, 4, 3)

# Get first matrix
first = tensor[0]  # Shape: (4, 3)

# Get first row of all matrices
first_rows = tensor[:, 0, :]  # Shape: (5, 3)

# Get specific element
element = tensor[2, 1, 0]  # Single value
```

---

#### Tensor Arithmetic

```python
import numpy as np

# Element-wise operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)  # [5 7 9]
print(a * b)  # [4 10 18]
print(a ** 2) # [1 4 9]

# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.dot(A, B))
# [[19 22]
#  [43 50]]
```

---

#### Broadcasting

**NumPy automatically expands dimensions for operations**

```python
# Add scalar to vector (broadcasts scalar)
vector = np.array([1, 2, 3])
result = vector + 10  # [11, 12, 13]

# Add vector to matrix (broadcasts vector)
matrix = np.array([[1, 2, 3], [4, 5, 6]])
vector = np.array([10, 20, 30])
result = matrix + vector
# [[11 22 33]
#  [14 25 36]]

# Add different shaped tensors
A = np.ones((3, 1))      # Shape: (3, 1)
B = np.ones((1, 4))      # Shape: (1, 4)
C = A + B                # Shape: (3, 4) - broadcasts both!
```

**[PLACEHOLDER FOR BROADCASTING VISUAL]**  
*Show:*
- *Small tensor being "stretched" to match larger tensor's shape*
- *Visual arrows showing dimension expansion*
- *Example: (3,1) + (1,4) → (3,4)*

---

## Working with Tensors

### NumPy (Foundation)

```python
import numpy as np

# Create tensors
zeros = np.zeros((3, 4))          # All zeros
ones = np.ones((2, 3))            # All ones
random = np.random.rand(2, 2)     # Random [0, 1)
arange = np.arange(10)            # [0, 1, 2, ..., 9]
linspace = np.linspace(0, 1, 5)   # 5 evenly spaced [0, 1]

# Operations
tensor = np.array([[1, 2], [3, 4]])
print(tensor.sum())              # 10 (sum all)
print(tensor.sum(axis=0))        # [4 6] (sum along axis 0)
print(tensor.mean())             # 2.5 (average)
print(tensor.max())              # 4 (maximum)
print(tensor.argmax())           # 3 (index of max)
```

---

### PyTorch (Deep Learning)

```python
import torch

# Create tensors
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3)
random = torch.rand(2, 2)
from_numpy = torch.from_numpy(np_array)

# GPU support
if torch.cuda.is_available():
    tensor_gpu = tensor.cuda()   # Move to GPU
    tensor_cpu = tensor_gpu.cpu() # Move back to CPU

# Autograd (automatic differentiation)
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()  # Compute gradient
print(x.grad)  # tensor([4.]) - dy/dx = 2x = 4

# Common operations
tensor = torch.tensor([[1, 2], [3, 4]])
print(tensor.sum())              # tensor(10)
print(tensor.mean())             # tensor(2.5)
print(tensor.transpose(0, 1))    # Transpose
```

---

### TensorFlow/Keras (Deep Learning)

```python
import tensorflow as tf

# Create tensors
zeros = tf.zeros([3, 4])
ones = tf.ones([2, 3])
random = tf.random.normal([2, 2])
from_numpy = tf.constant(np_array)

# Operations
tensor = tf.constant([[1, 2], [3, 4]])
print(tf.reduce_sum(tensor))     # 10
print(tf.reduce_mean(tensor))    # 2.5
print(tf.transpose(tensor))      # Transpose

# Variables (trainable)
weight = tf.Variable(tf.random.normal([784, 10]))
# Can be updated during training
```

---

## Real-World Applications

### Application 1: Image Classification

```python
# CNN for image classification
import torch
import torch.nn as nn

# Input: Batch of images
input_tensor = torch.rand(32, 3, 224, 224)  # 32 images, RGB, 224×224

# Pass through CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)   # Input: (32, 3, 224, 224)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 111 * 111, 10)
    
    def forward(self, x):
        x = self.conv1(x)      # Shape: (32, 64, 222, 222)
        x = self.pool(x)       # Shape: (32, 64, 111, 111)
        x = x.view(-1, 64*111*111)  # Flatten: (32, 788544)
        x = self.fc(x)         # Shape: (32, 10)
        return x

model = SimpleCNN()
output = model(input_tensor)  # Shape: (32, 10) - 10 class probabilities for each image
```

---

### Application 2: Text Processing

```python
# RNN for text classification
# Input: Batch of sentences (words represented as embeddings)

# Shape: (batch_size, sequence_length, embedding_dim)
text_tensor = torch.rand(16, 50, 300)  # 16 sentences, 50 words each, 300-dim embeddings

import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(300, 128, batch_first=True)  # 300 input, 128 hidden
        self.fc = nn.Linear(128, 2)  # 2 classes (positive/negative)
    
    def forward(self, x):
        # x shape: (16, 50, 300)
        out, (hidden, cell) = self.rnn(x)
        # hidden shape: (1, 16, 128)
        last_hidden = hidden[-1]  # (16, 128)
        output = self.fc(last_hidden)  # (16, 2)
        return output

model = SimpleRNN()
output = model(text_tensor)  # Shape: (16, 2)
```

---

### Application 3: Time Series Prediction

```python
# Predicting future values from past sequences
# Input: Historical data

# Shape: (batch_size, timesteps, features)
time_series = torch.rand(64, 100, 5)  # 64 sequences, 100 timesteps, 5 features

# Process with RNN/LSTM to predict next timestep
# Output shape: (64, 1, 5) - next timestep prediction
```

---

## Tensor Shape Cheat Sheet

```
Common Shapes in ML:

Scalar (0D):      ()
Vector (1D):      (n,)
Matrix (2D):      (rows, cols)

Images:
Grayscale:        (height, width)
RGB:              (height, width, 3)
Batch (TF):       (batch, height, width, channels)
Batch (PyTorch):  (batch, channels, height, width)

Sequences:
Single:           (timesteps, features)
Batch:            (batch, timesteps, features)

Videos:
Single:           (frames, height, width, channels)
Batch:            (batch, frames, height, width, channels)

Neural Network:
Dense weights:    (input_size, output_size)
Conv filters:     (out_channels, in_channels, kernel_h, kernel_w)
```

---

## 🧠 Quick Quiz

1. What's the rank of a grayscale image tensor with shape (28, 28)?
2. What does a tensor shape of (32, 224, 224, 3) represent?
3. How many elements are in a tensor with shape (5, 4, 3)?
4. What's the difference between a 1D tensor with shape (5,) and a 2D tensor with shape (5, 1)?
5. In PyTorch, what's the typical shape format for a batch of images?

<details>
<summary>Click for answers</summary>

1. Rank 2 (2D tensor - it has 2 axes: height and width)
2. A batch of 32 RGB images, each 224×224 pixels (format: batch, height, width, channels)
3. 5 × 4 × 3 = 60 elements
4. (5,) is a 1D vector, (5, 1) is a 2D matrix with 5 rows and 1 column. Both have 5 elements but different ranks.
5. (batch_size, channels, height, width) - channels come before spatial dimensions in PyTorch

</details>

---

## Summary

🎯 **Key Takeaways**:

**Tensor Basics**:
- Tensor = multi-dimensional array of numbers
- Rank = number of dimensions (0D, 1D, 2D, 3D, ...)
- Shape = size along each dimension
- All data in deep learning → tensors

**Common Ranks**:
- 0D (Scalar): Single number (loss, accuracy)
- 1D (Vector): Features, embeddings, sequences
- 2D (Matrix): Datasets, grayscale images
- 3D: RGB images, time series batches
- 4D: Batches of images, videos
- 5D: Batches of videos

**Remember**: 
- Understanding tensor shapes is crucial for debugging
- Most errors in deep learning = shape mismatches
- Always check `.shape` when debugging!

**Practical Tip**: When working with tensors, always print the shape to understand data dimensions!

---

*Previous: [← Job Roles in ML](./08_job_roles_in_ml.md)*  
*Next: [Mathematics for ML →](./10_mathematics_for_ml.md)*
