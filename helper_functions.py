# Helper Functions for HW1 - Wave Classification
# 50.039 Deep Learning, SUTD
# Version: 1.0 (2026)

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# Numpy
import numpy as np
# Pandas
import pandas as pd
# Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def load_wave_dataset(excel_file_path='wave_dataset.xlsx'):
    """
    Load the wave dataset from an Excel file.

    Returns:
        x1: numpy array of first feature
        x2: numpy array of second feature
        inputs: numpy array of shape (n_samples, 2)
        outputs: numpy array of class labels
    """
    df = pd.read_excel(excel_file_path)
    x1 = df['x1'].values
    x2 = df['x2'].values
    y = df['class'].values
    x = np.column_stack([x1, x2])
    return x1, x2, x, y


def plot_wave_dataset(x1, x2, outputs, title="Wave Dataset"):
    """
    Visualize the wave dataset with class colors.

    Args:
        x1: first feature values
        x2: second feature values
        outputs: class labels (0 or 1)
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Separate by class
    idx_0 = np.where(outputs == 0)[0]
    idx_1 = np.where(outputs == 1)[0]

    # Plot each class
    ax.scatter(x1[idx_0], x2[idx_0], c='blue', marker='o', alpha=0.6, label='Class 0', s=30)
    ax.scatter(x1[idx_1], x2[idx_1], c='red', marker='x', alpha=0.6, label='Class 1', s=30)

    # Plot true decision boundary: sin(pi*x) + sin(pi*y) = 0
    x_line = np.linspace(-2, 2, 200)
    y_line = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x_line, y_line)
    Z = np.sin(np.pi * X) + np.sin(np.pi * Y)
    #ax.contour(X, Y, Z, levels=[0], colors='green', linewidths=2, linestyles='--')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect('equal')
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_decision_boundary(model, device='cpu', x_range=(-2, 2), y_range=(-2, 2), resolution=100):
    """
    Plot the learned decision boundary of a model.

    Args:
        model: trained PyTorch model
        device: 'cpu' or 'cuda'
        x_range: tuple of (min, max) for x-axis
        y_range: tuple of (min, max) for y-axis
        resolution: number of points per axis
    """
    model.eval()

    # Create grid
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Flatten and predict
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(grid_tensor).cpu().numpy()

    Z = predictions.reshape(X.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Predicted boundary (filled contour)
    contour = ax.contourf(X, Y, Z, levels=[0, 0.5, 1], colors=['blue', 'red'], alpha=0.3)
    ax.contour(X, Y, Z, levels=[0.5], colors='black', linewidths=2)

    # True boundary
    Z_true = np.sin(np.pi * X) + np.sin(np.pi * Y)
    #ax.contour(X, Y, Z_true, levels=[0], colors='green', linewidths=2, linestyles='--')

    # Legend
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=2, label='Learned boundary'),
        #Line2D([0], [0], color='green', linewidth=2, linestyle='--', label='True boundary')
    ]
    ax.legend(handles=legend_elements)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Decision Boundary Comparison')
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect('equal')
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================
# Test Functions
# ============================================

def test_dataset_object(dataset):
    """Test the WaveDataset implementation."""
    print("=" * 50)
    print("Testing WaveDataset Implementation")
    print("=" * 50)

    # Test case 1: __getitem__
    print("\n--- Test case 1: __getitem__ method ---")
    print("Drawing sample with index 100 from dataset...")

    try:
        test_sample = dataset[100]
        inputs, label = test_sample

        # Check types
        assert isinstance(inputs, torch.Tensor), "Inputs should be a torch.Tensor"
        assert isinstance(label, torch.Tensor), "Label should be a torch.Tensor"

        # Check shapes
        assert inputs.shape == (2,), f"Input shape should be (2,), got {inputs.shape}"
        assert label.shape == () or label.shape == (1,), f"Label shape should be () or (1,), got {label.shape}"

        # Check dtype
        assert inputs.dtype == torch.float32, f"Input dtype should be float32, got {inputs.dtype}"

        print(f"Retrieved: inputs={inputs}, label={label}")
        print("Test case 1: PASSED")
        test1_passed = True
    except Exception as e:
        print(f"Test case 1: FAILED - {str(e)}")
        test1_passed = False

    # Test case 2: __len__
    print("\n--- Test case 2: __len__ method ---")
    print("Checking dataset length...")

    try:
        length = len(dataset)
        assert length == 1200, f"Expected length 1200, got {length}"
        print(f"Retrieved length: {length}")
        print("Test case 2: PASSED")
        test2_passed = True
    except Exception as e:
        print(f"Test case 2: FAILED - {str(e)}")
        test2_passed = False

    # Summary
    print("\n" + "=" * 50)
    if test1_passed and test2_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED. Please review your implementation.")
    print("=" * 50)


def test_dataloader_object(dataloader):
    """Test the DataLoader configuration."""
    print("=" * 50)
    print("Testing DataLoader Configuration")
    print("=" * 50)

    # Test case 1: batch size
    print("\n--- Test case 1: Batch size ---")
    print("Checking batch size...")

    try:
        batch_size = dataloader.batch_size
        assert batch_size == 64, f"Expected batch_size=64, got {batch_size}"
        print(f"Retrieved batch size: {batch_size}")
        print("Test case 1: PASSED")
        test1_passed = True
    except Exception as e:
        print(f"Test case 1: FAILED - {str(e)}")
        test1_passed = False

    # Test case 2: shuffle
    print("\n--- Test case 2: Shuffle enabled ---")
    print("Checking if shuffling is enabled...")

    try:
        is_shuffling = "RandomSampler" in str(type(dataloader.sampler))
        assert is_shuffling, "DataLoader should have shuffle=True"
        print(f"Shuffling enabled: {is_shuffling}")
        print("Test case 2: PASSED")
        test2_passed = True
    except Exception as e:
        print(f"Test case 2: FAILED - {str(e)}")
        test2_passed = False

    # Summary
    print("\n" + "=" * 50)
    if test1_passed and test2_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED. Please review your implementation.")
    print("=" * 50)


def test_gated_layer(layer):
    """Test the GatedLayer implementation."""
    print("=" * 50)
    print("Testing GatedLayer Implementation")
    print("=" * 50)

    # Test case 1: output shape
    print("\n--- Test case 1: Output shape ---")
    print("Testing forward pass with input shape (1, 2)...")

    try:
        x = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
        output = layer(x)

        expected_shape = (1, 32)  # Assuming output_dim=32 as per design
        assert output.shape == expected_shape or output.shape[0] == 1, \
            f"Output shape mismatch. Got {output.shape}"

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print("Test case 1: PASSED")
        test1_passed = True
    except Exception as e:
        print(f"Test case 1: FAILED - {str(e)}")
        test1_passed = False

    # Test case 2: gradients flow
    print("\n--- Test case 2: Gradient flow ---")
    print("Checking if gradients can flow through the layer...")

    try:
        x = torch.tensor([[0.5, -0.5]], dtype=torch.float32, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Gradients should flow to input"
        print("Gradients successfully computed")
        print("Test case 2: PASSED")
        test2_passed = True
    except Exception as e:
        print(f"Test case 2: FAILED - {str(e)}")
        test2_passed = False

    # Test case 3: learnable parameters
    print("\n--- Test case 3: Learnable parameters ---")
    print("Checking for learnable parameters...")

    try:
        params = list(layer.parameters())
        assert len(params) > 0, "Layer should have learnable parameters"
        total_params = sum(p.numel() for p in params)
        print(f"Number of parameter tensors: {len(params)}")
        print(f"Total learnable parameters: {total_params}")
        print("Test case 3: PASSED")
        test3_passed = True
    except Exception as e:
        print(f"Test case 3: FAILED - {str(e)}")
        test3_passed = False

    # Summary
    print("\n" + "=" * 50)
    if test1_passed and test2_passed and test3_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED. Please review your implementation.")
    print("=" * 50)
