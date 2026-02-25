# HW1 - Deep Learning (50.039 SUTD)
## Wave Classification with Gated Neural Networks

**Course:** 50.039 Deep Learning
**Institution:** Singapore University of Technology and Design
**Author:** Matthieu DE MARI
**Year:** 2026

---

## Overview

This homework consists of **3 Jupyter notebooks** with a total of **18 questions**. You will build a neural network to classify a wave interference pattern dataset.

### Learning Objectives
- Implement a PyTorch `Dataset` and `DataLoader`
- Build a custom **Gated Linear Unit (GLU)** layer
- Train and evaluate a neural network classifier
- Analyze model generalization (overfitting/underfitting)

---

## File Structure

```
HW1-2026/
├── HW1-A. Wave Data and DataLoader.ipynb    # Part A (Q1-Q6)
├── HW1-B. A Gated Activation Function.ipynb # Part B (Q7-Q10)
├── HW1-C. Building and Training the Model.ipynb # Part C (Q11-Q18)
├── helper_functions.py                      # Helper functions (DO NOT MODIFY)
├── wave_dataset.xlsx                        # Training data (1200 samples)
├── unseen_wave_dataset.xlsx                 # Test data (for evaluation)
└── Solution/                                # Reference solutions (TA only)
```

---

## Requirements

```
Python 3
Matplotlib
NumPy
Pandas
PyTorch
Torchmetrics
```

---

## Part A: Wave Data and DataLoader (Q1-Q6)

**File:** `HW1-A. Wave Data and DataLoader.ipynb`

### Task Summary
1. Understand the wave interference dataset
2. Implement `WaveDataset` class (PyTorch Dataset)
3. Create a `DataLoader` with batch size 64 and shuffling

### Questions

| Q# | Type | Topic |
|----|------|-------|
| Q1 | Written | Describe the ML problem (Task, Dataset, Inputs, Outputs) |
| Q2 | Written | Explain the decision boundary geometry |
| Q3 | Code | Show completed `WaveDataset` class |
| Q4 | Written | Conceptual questions about Dataset class |
| Q5 | Code | Show completed `DataLoader` creation |
| Q6 | Written | Conceptual questions about DataLoaders |

### Key Concepts
- Decision boundary: $\sin(\pi x_1) + \sin(\pi x_2) = 0$
- Binary classification: Class 0 vs Class 1
- Dataset methods: `__init__`, `__len__`, `__getitem__`

---

## Part B: Gated Activation Function (Q7-Q10)

**File:** `HW1-B. A Gated Activation Function.ipynb`

### Task Summary
Implement a **Gated Linear Unit (GLU)** layer:

$$f(x) = \text{Linear}_1(x) \odot \sigma(\text{Linear}_2(x))$$

- **Value path:** $W_1 x + b_1$ (the information)
- **Gate path:** $\sigma(W_2 x + b_2)$ (how much to pass, 0-1)

### Questions

| Q# | Type | Topic |
|----|------|-------|
| Q7 | Written | Analyze gate behavior (0, 0.5, 1 values) |
| Q8 | Written | Differentiability and gradient flow |
| Q9 | Code | Show completed `GatedLayer` class |
| Q10 | Written | Compare with Linear+ReLU |

---

## Part C: Building and Training the Model (Q11-Q18)

**File:** `HW1-C. Building and Training the Model.ipynb`

### Task Summary
1. Build `WaveClassifier` neural network
2. Implement training loop
3. Evaluate on unseen test data
4. Analyze generalization

### Network Architecture
```
Input (2) → GatedLayer (32) → Linear+ReLU (64) → Linear+ReLU (32) → Linear+Sigmoid (1)
```

### Questions

| Q# | Type | Topic |
|----|------|-------|
| Q11 | Code | Show completed `WaveClassifier` class |
| Q12 | Written | Explain Sigmoid activation for binary classification |
| Q13 | Written | Explain BCE loss function |
| Q14 | Code | Complete training loop, report accuracy (>90%) |
| Q15 | Written | Explain `optimizer.zero_grad()` |
| Q16 | Code | Write evaluation code, report test accuracy |
| Q17 | Written | Analyze overfitting/underfitting |
| Q18 | Written | Name 2 regularization techniques |

### Training Parameters
- Epochs: 100
- Batch size: 64
- Learning rate: 0.01 or 0.001 (experiment)
- Optimizer: Adam
- Loss: BCELoss

---

## Submission Checklist

### Code Questions (show in report)
- [ ] Q3: `WaveDataset` class
- [ ] Q5: `DataLoader` creation
- [ ] Q9: `GatedLayer` class
- [ ] Q11: `WaveClassifier` class
- [ ] Q14: Training loop + final accuracy
- [ ] Q16: Test evaluation code + accuracy

### Written Questions
- [ ] Q1: ML problem description
- [ ] Q2: Decision boundary geometry
- [ ] Q4: Dataset conceptual questions
- [ ] Q6: DataLoader conceptual questions
- [ ] Q7: Gate behavior analysis
- [ ] Q8: Differentiability and gradients
- [ ] Q10: Comparison with Linear+ReLU
- [ ] Q12: Sigmoid activation
- [ ] Q13: BCE loss formula
- [ ] Q15: `optimizer.zero_grad()` explanation
- [ ] Q17: Overfitting/underfitting analysis
- [ ] Q18: Two regularization techniques

---

## Tips

1. **Run cells in order** - Each notebook builds on previous cells
2. **Use test functions** - `test_dataset_object()`, `test_dataloader_object()`, `test_gated_layer()` verify your implementations
3. **Copy code forward** - Part C requires your completed classes from Parts A and B
4. **GPU optional** - CPU is sufficient for this homework
5. **Target accuracy** - Training accuracy should exceed 90%

---

## Workflow

```
1. Complete HW1-A → Get WaveDataset and DataLoader working
                  ↓
2. Complete HW1-B → Get GatedLayer working
                  ↓
3. Complete HW1-C → Copy classes from A & B
                  → Build WaveClassifier
                  → Train and evaluate
                  → Answer analysis questions
```

Good luck!
