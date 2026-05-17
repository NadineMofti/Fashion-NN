# Fashion-MNIST Classification using PyTorch

A Deep Learning project using PyTorch to classify Fashion-MNIST images using Multi-Layer Perceptron (MLP) models.

---

# Project Overview

This project compares two different neural network architectures using:

- ReLU Activation Function
- Tanh Activation Function

The project includes:
- Data preprocessing
- Data augmentation
- Neural network training
- Model evaluation
- Performance visualization

---

# Dataset

The project uses the Fashion-MNIST dataset provided by torchvision.

Dataset Details:
- 60,000 Training Images
- 10,000 Testing Images
- Image Size: 28×28
- 10 Classes

Classes include:
- T-shirt
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

---

# Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- Torchvision

---

# Project Structure

```bash
project/
│
├── data/
├── notebook.ipynb
├── README.md
```

---

# Data Preprocessing

The preprocessing pipeline includes:

- Checking missing values
- Splitting dataset into:
  - Training Set
  - Validation Set
  - Testing Set
- Feature scaling using StandardScaler
- Converting data into PyTorch tensors

---

# Data Augmentation

Gaussian noise is added to training data to:
- improve generalization
- reduce overfitting
- increase dataset diversity

---

# Model Architectures

## Model 1 — ReLU

```python
nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(128, 10)
)
```

---

## Model 2 — Tanh

```python
nn.Sequential(
    nn.Linear(784, 64),
    nn.BatchNorm1d(64),
    nn.Tanh(),
    nn.Dropout(0.3),

    nn.Linear(64, 32),
    nn.BatchNorm1d(32),
    nn.Tanh(),
    nn.Dropout(0.3),

    nn.Linear(32, 10)
)
```

---

# Training Configuration

## Optimizers

- Adam Optimizer

## Learning Rates

| Model | Learning Rate |
|------|------|
| ReLU Model | 0.0001 |
| Tanh Model | 0.001 |

---

# Loss Function

```python
nn.CrossEntropyLoss()
```

Used for multi-class classification tasks.

---

# Scheduler

```python
StepLR(step_size=10, gamma=0.1)
```

Used to reduce learning rate during training.

---

# Training

Models are trained for:

```python
50 Epochs
```

The training process includes:
- Forward propagation
- Loss computation
- Backpropagation
- Validation evaluation

---

# Evaluation

The models are evaluated using:

- Test Accuracy
- Validation Loss
- Training Loss

---

# Visualization

The project visualizes:
- Training Loss
- Validation Loss
- Model Comparison
- Final Accuracy Comparison

Using Matplotlib.

---

# Results

The project compares:
- ReLU vs Tanh performance
- Accuracy differences
- Validation loss behavior

---

# How to Run

## Install Requirements

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

---

## Run the Notebook

```bash
jupyter notebook
```

Open the notebook and run all cells.

---

# Future Improvements

- Add CNN models
- Use GPU training
- Hyperparameter tuning
- Add confusion matrix
- Improve augmentation techniques

---

# Author

Developed using PyTorch for Deep Learning experimentation.
