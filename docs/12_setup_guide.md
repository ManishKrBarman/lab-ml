# Chapter 12: Development Environment Setup

## 📖 Table of Contents
- [Introduction](#introduction)
- [Python Installation](#python-installation)
- [Package Managers](#package-managers)
- [Virtual Environments](#virtual-environments)
- [IDEs and Editors](#ides-and-editors)
- [Jupyter Notebooks](#jupyter-notebooks)
- [GPU Setup](#gpu-setup)
- [Cloud Development](#cloud-development)
- [Essential Tools](#essential-tools)

---

## Introduction

Setting up a proper development environment is crucial for machine learning work. This guide will help you get everything configured correctly.

### What You'll Need

```
Core Setup:
├─ Python 3.8+ (Programming language)
├─ pip/conda (Package manager)
├─ Virtual environment (Isolation)
├─ IDE/Editor (Code writing)
└─ Jupyter (Interactive coding)

Optional (Advanced):
├─ GPU drivers (CUDA for deep learning)
├─ Docker (Containerization)
├─ Git (Version control)
└─ Cloud platforms (Scalable computing)
```

---

## Python Installation

### Windows

**Method 1: Official Python**
1. Download from [python.org](https://www.python.org/downloads/)
2. Run installer
3. ✅ Check "Add Python to PATH"
4. Click "Install Now"

**Verify installation**:
```bash
python --version
# Python 3.11.x

pip --version
# pip 23.x.x
```

**Method 2: Anaconda** (Recommended for ML)
1. Download from [anaconda.com](https://www.anaconda.com/products/distribution)
2. Run installer
3. Choose "Just Me"
4. Add to PATH (optional)

```bash
conda --version
# conda 23.x.x
```

---

### macOS

**Method 1: Homebrew**
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Verify
python3 --version
pip3 --version
```

**Method 2: Official Python**
- Download from [python.org](https://www.python.org/downloads/mac-osx/)
- Run .pkg installer

**Method 3: Anaconda**
- Download from [anaconda.com](https://www.anaconda.com/products/distribution)
- Run installer

---

### Linux (Ubuntu/Debian)

```bash
# Update system
sudo apt update
sudo apt upgrade

# Install Python and pip
sudo apt install python3 python3-pip

# Verify
python3 --version
pip3 --version

# Install development tools
sudo apt install python3-dev python3-venv build-essential
```

---

## Package Managers

### pip (Python Package Installer)

**Basic Commands**:
```bash
# Install package
pip install numpy

# Install specific version
pip install numpy==1.24.0

# Install multiple packages
pip install numpy pandas matplotlib

# Install from requirements.txt
pip install -r requirements.txt

# Upgrade package
pip install --upgrade numpy

# Uninstall package
pip uninstall numpy

# List installed packages
pip list

# Show package info
pip show numpy

# Search for package
pip search sklearn
```

**Create requirements.txt**:
```bash
# Save current environment
pip freeze > requirements.txt

# Or manually create:
# requirements.txt
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.3.0
matplotlib==3.7.0
```

---

### conda (Anaconda/Miniconda)

**Why conda?**
- Manages Python AND non-Python dependencies
- Better environment isolation
- Pre-compiled binaries (faster installation)
- Handles complex dependencies (e.g., CUDA)

**Basic Commands**:
```bash
# Install package
conda install numpy

# Install from specific channel
conda install -c conda-forge xgboost

# Install multiple packages
conda install numpy pandas matplotlib

# Update package
conda update numpy

# Remove package
conda remove numpy

# List installed packages
conda list

# Search for package
conda search tensorflow
```

**Export environment**:
```bash
# Create environment file
conda env export > environment.yml

# Install from environment file
conda env create -f environment.yml
```

---

## Virtual Environments

### Why Virtual Environments?

**Problem**: Different projects need different package versions

```
Project A: tensorflow==2.10.0
Project B: tensorflow==2.15.0
❌ Conflict!
```

**Solution**: Isolated environments

```
project_a_env: tensorflow==2.10.0
project_b_env: tensorflow==2.15.0
✅ No conflict!
```

---

### venv (Built-in Python)

**Create environment**:
```bash
# Create virtual environment
python -m venv myenv

# Or specify Python version
python3.11 -m venv myenv
```

**Activate environment**:
```bash
# Windows
myenv\Scripts\activate

# macOS/Linux
source myenv/bin/activate

# You'll see (myenv) in terminal prompt
```

**Deactivate**:
```bash
deactivate
```

**Complete workflow**:
```bash
# Create project directory
mkdir ml_project
cd ml_project

# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# Install packages
pip install numpy pandas scikit-learn

# Save dependencies
pip freeze > requirements.txt

# Work on project...

# Deactivate when done
deactivate
```

---

### conda environments

**Create environment**:
```bash
# Create environment with Python version
conda create -n myenv python=3.11

# Create with packages
conda create -n ml_env python=3.11 numpy pandas scikit-learn

# Create from environment file
conda env create -f environment.yml
```

**Activate/Deactivate**:
```bash
# Activate
conda activate ml_env

# Deactivate
conda deactivate
```

**Manage environments**:
```bash
# List environments
conda env list

# Remove environment
conda env remove -n myenv

# Clone environment
conda create --name new_env --clone old_env
```

**Example workflow**:
```bash
# Create ML environment
conda create -n ml_project python=3.11

# Activate
conda activate ml_project

# Install ML packages
conda install numpy pandas scikit-learn matplotlib jupyter

# Install deep learning
conda install tensorflow  # or pytorch

# Export environment
conda env export > environment.yml

# Deactivate
conda deactivate
```

---

## IDEs and Editors

### VS Code (Recommended)

**Why VS Code?**
- Free and open-source
- Excellent Python support
- Jupyter integration
- Git integration
- Extensions ecosystem

**Installation**:
1. Download from [code.visualstudio.com](https://code.visualstudio.com/)
2. Install for your OS

**Essential Extensions**:
```
1. Python (Microsoft) - Core Python support
2. Pylance - Advanced Python language support
3. Jupyter - Notebook support
4. Python Debugger - Debugging
5. autoDocstring - Generate docstrings
6. GitLens - Enhanced Git
```

**Setup**:
```json
// settings.json
{
    "python.defaultInterpreterPath": "/path/to/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "jupyter.askForKernelRestart": false
}
```

**Keyboard Shortcuts**:
```
Ctrl/Cmd + Shift + P: Command palette
Ctrl/Cmd + `: Terminal
F5: Run debugger
Shift + Enter: Run cell (in notebook)
Ctrl/Cmd + /: Comment line
```

---

### PyCharm

**Professional IDE for Python**

**Versions**:
- **Community**: Free, good for general Python
- **Professional**: Paid, includes scientific tools, Jupyter, remote development

**Pros**:
- Powerful debugging
- Smart code completion
- Refactoring tools
- Database tools

**Cons**:
- Heavy (uses more resources)
- Paid for full features

---

### Jupyter Lab/Notebook

**Best for**:
- Interactive data exploration
- Visualization
- Experimentation
- Sharing results

**Installation**:
```bash
pip install jupyterlab notebook

# Or with conda
conda install jupyterlab notebook
```

**Launch**:
```bash
# Jupyter Notebook
jupyter notebook

# Jupyter Lab (modern interface)
jupyter lab

# Opens in browser at http://localhost:8888
```

**[PLACEHOLDER FOR JUPYTER INTERFACE]**  
*Screenshot showing:*
- *Jupyter Lab interface*
- *Code cell, Markdown cell*
- *Output with plots*
- *File browser sidebar*

---

### Google Colab

**Free cloud-based Jupyter**

**Pros**:
- Free GPU/TPU
- No setup required
- Easy sharing
- Google Drive integration

**Cons**:
- Session limits
- Internet required
- Less customization

**Access**: [colab.research.google.com](https://colab.research.google.com/)

**Quick Start**:
```python
# In Colab cell:

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install packages
!pip install xgboost

# Use GPU
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## Jupyter Notebooks

### Notebook Basics

**Cell Types**:
```python
# Code Cell
import numpy as np
x = np.array([1, 2, 3])
print(x)
```

```markdown
# Markdown Cell
## Heading
**Bold text**
*Italic text*
- Bullet point
```

**Magic Commands**:
```python
# Time execution
%timeit sum(range(1000))

# Run external script
%run script.py

# List variables
%whos

# Matplotlib inline
%matplotlib inline

# Load code from file
%load script.py

# System commands
!ls  # List files
!pip install numpy
```

---

### Best Practices

**1. Organize Your Notebook**:
```markdown
# Project Title

## 1. Setup
- Imports
- Configuration

## 2. Load Data
- Data loading
- Initial exploration

## 3. EDA
- Visualizations
- Statistics

## 4. Preprocessing
- Cleaning
- Feature engineering

## 5. Modeling
- Training
- Evaluation

## 6. Conclusion
- Summary
- Next steps
```

**2. Keep Cells Small**:
```python
# ❌ Bad: One giant cell
import numpy as np
import pandas as pd
df = pd.read_csv('data.csv')
df = df.dropna()
df['new_col'] = df['col1'] + df['col2']
# ... 100 more lines

# ✅ Good: Logical sections
# Cell 1: Imports
import numpy as np
import pandas as pd

# Cell 2: Load data
df = pd.read_csv('data.csv')

# Cell 3: Clean data
df = df.dropna()

# Cell 4: Feature engineering
df['new_col'] = df['col1'] + df['col2']
```

**3. Add Markdown Explanations**:
```markdown
## Data Cleaning

We remove rows with missing values because:
1. Missing data is < 1% of dataset
2. Values are missing at random
3. Imputation would introduce bias
```

---

## GPU Setup

### Why GPU?

**CPU vs GPU**:
```
Task: Train neural network on 10,000 images

CPU: 2-3 hours
GPU: 10-15 minutes

Speedup: 10-20x!
```

---

### Check GPU Availability

**PyTorch**:
```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

**TensorFlow**:
```python
import tensorflow as tf

print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
```

---

### NVIDIA Setup (CUDA)

**Requirements**:
1. NVIDIA GPU (GTX/RTX series)
2. CUDA drivers
3. cuDNN library

**Installation (Windows)**:
1. Download CUDA Toolkit: [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
2. Download cuDNN: [developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)
3. Install CUDA first, then cuDNN
4. Verify:
```bash
nvidia-smi  # Should show GPU info
```

**Installation (Linux)**:
```bash
# Ubuntu
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Install CUDA
sudo apt update
sudo apt install cuda

# Verify
nvidia-smi
```

**Install GPU-enabled PyTorch**:
```bash
# Visit pytorch.org for latest command
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Install GPU-enabled TensorFlow**:
```bash
pip install tensorflow[and-cuda]
# Or
conda install tensorflow-gpu
```

---

## Cloud Development

### Google Colab

**Free tier**:
- Free GPU (Tesla K80/T4)
- 12 hours session limit
- Limited RAM

**Pro ($9.99/month)**:
- Better GPUs
- Longer sessions
- More RAM

**Usage**:
```python
# Check GPU
!nvidia-smi

# Install packages
!pip install transformers

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Upload files
from google.colab import files
uploaded = files.upload()
```

---

### Kaggle Notebooks

**Free tier**:
- Free GPU (Tesla P100)
- 30 hours/week GPU
- 9 hours session

**Pros**:
- Access to Kaggle datasets
- Competition submission
- Community kernels

**Access**: [kaggle.com/code](https://www.kaggle.com/code)

---

### AWS SageMaker Studio Lab

**Free tier**:
- 12 hours session
- Free GPU (limited hours)
- Persistent storage

**Access**: [studiolab.sagemaker.aws](https://studiolab.sagemaker.aws/)

---

### Comparison

| Platform | GPU | Free Tier | Pro Tier | Best For |
|----------|-----|-----------|----------|----------|
| **Google Colab** | T4/K80 | 12hrs/session | $10/mo | Quick experiments |
| **Kaggle** | P100 | 30hrs/week | N/A | Competitions |
| **SageMaker Lab** | T4 | Limited | N/A | Learning |
| **Paperspace** | Various | Limited | From $8/mo | Serious projects |

---

## Essential Tools

### Version Control (Git)

**Installation**:
```bash
# Windows: Download from git-scm.com
# macOS: brew install git
# Linux: sudo apt install git

# Configure
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

**Basic Workflow**:
```bash
# Initialize repository
git init

# Add files
git add .

# Commit
git commit -m "Initial commit"

# Create GitHub repo and push
git remote add origin https://github.com/username/repo.git
git push -u origin main
```

**For ML Projects**:
```bash
# .gitignore
data/
*.csv
*.h5
*.pkl
__pycache__/
.ipynb_checkpoints/
.env
```

---

### Docker (Optional)

**Why Docker?**
- Reproducible environments
- Easy deployment
- Consistent across systems

**Basic Dockerfile for ML**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "train.py"]
```

**Build and run**:
```bash
# Build image
docker build -t ml-project .

# Run container
docker run ml-project
```

---

### MLflow (Experiment Tracking)

**Installation**:
```bash
pip install mlflow
```

**Usage**:
```python
import mlflow

# Start experiment
mlflow.set_experiment("my_experiment")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 100)
    
    # Train model...
    # model.fit(...)
    
    # Log metrics
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("loss", 0.23)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

**View results**:
```bash
mlflow ui
# Opens at http://localhost:5000
```

---

## Complete Setup Checklist

### Beginner Setup
```
✅ Install Python 3.8+
✅ Install pip/conda
✅ Create virtual environment
✅ Install: numpy, pandas, matplotlib, scikit-learn
✅ Install Jupyter
✅ Install VS Code (optional but recommended)
✅ Test installation
```

**Test Script**:
```python
# test_setup.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print("✅ NumPy version:", np.__version__)
print("✅ Pandas version:", pd.__version__)
print("✅ Matplotlib version:", plt.matplotlib.__version__)
print("✅ Scikit-learn imported successfully")

# Quick test
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])
model = LinearRegression()
model.fit(X, y)
print("✅ Model training works!")
print(f"   Prediction for X=4: {model.predict([[4]])[0]:.1f}")
```

---

### Advanced Setup (Deep Learning)
```
✅ All from beginner setup
✅ Install TensorFlow OR PyTorch
✅ Setup GPU (CUDA + cuDNN)
✅ Test GPU availability
✅ Install additional libraries (transformers, opencv, etc.)
✅ Setup cloud access (Colab, Kaggle)
✅ Install Git
✅ Setup Docker (optional)
```

**Test GPU Script**:
```python
# test_gpu.py
import tensorflow as tf
import torch

print("TensorFlow GPUs:", tf.config.list_physical_devices('GPU'))
print("PyTorch CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print(f"PyTorch GPU: {torch.cuda.get_device_name(0)}")
```

---

## Troubleshooting

### Common Issues

**1. "Python not found"**
```bash
# Add Python to PATH
# Windows: Search "Environment Variables" → Edit PATH
# Add: C:\Users\YourName\AppData\Local\Programs\Python\Python311
```

**2. "pip not found"**
```bash
# Reinstall pip
python -m ensurepip --upgrade
```

**3. "Module not found"**
```bash
# Make sure virtual environment is activated
# Check: which python (Linux/Mac) or where python (Windows)
# Should point to virtual environment

# Reinstall package
pip install package_name
```

**4. "CUDA not available"**
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall CUDA-enabled package
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**5. Jupyter kernel issues**
```bash
# Install ipykernel in virtual environment
pip install ipykernel

# Add kernel to Jupyter
python -m ipykernel install --user --name=myenv --display-name="Python (myenv)"
```

---

## Summary

🎯 **Key Takeaways**:

**Essential Setup**:
1. **Python 3.8+**: Core language
2. **Virtual environments**: Isolate projects
3. **Package manager**: pip or conda
4. **IDE**: VS Code recommended
5. **Jupyter**: Interactive development

**Best Practices**:
- Always use virtual environments
- Keep requirements.txt updated
- Use version control (Git)
- Test setup with sample code
- Start with CPU, upgrade to GPU when needed

**Quick Start Command**:
```bash
# Create new ML project
mkdir my_ml_project
cd my_ml_project
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter lab
```

**Remember**: Perfect setup takes time. Start simple, add tools as you need them!

---

*Previous: [← ML Frameworks & Libraries](./11_ml_frameworks.md)*  
*Back to: [Main README](./README.md)*

---

## 🎉 Congratulations!

You've completed the Machine Learning Learning Path! You now have:
- ✅ Understanding of ML fundamentals
- ✅ Knowledge of different ML types and approaches
- ✅ Awareness of real-world applications
- ✅ Grasp of ML development lifecycle
- ✅ Career path options
- ✅ Mathematical foundations
- ✅ Familiarity with ML tools and frameworks
- ✅ Fully configured development environment

**Next Steps**:
1. **Practice**: Work on small projects
2. **Kaggle**: Join competitions
3. **Build Portfolio**: Create GitHub projects
4. **Learn by Doing**: Apply concepts to real problems
5. **Stay Updated**: Follow ML research and trends

**Happy Learning! 🚀**
