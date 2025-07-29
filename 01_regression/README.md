# Lesson 01: Linear Regression with PyTorch

## Overview
This lesson covers the fundamentals of linear regression using PyTorch, progressing from basic concepts to a complete implementation with device-agnostic code.

## What You'll Learn
- Creating synthetic linear data with known parameters
- Building custom linear regression models using `nn.Module`
- Understanding the training loop fundamentals
- Implementing device-agnostic code (CPU/GPU compatibility)
- Model saving and loading best practices
- Data visualization with matplotlib

## Key Concepts Covered

### 1. Linear Regression Model Architecture
- Simple linear transformation: `y = weight * x + bias`
- Using `nn.Linear()` for linear transformations
- Understanding model parameters and gradients

### 2. Training Loop Components
- Forward pass: making predictions
- Loss calculation using Mean Squared Error (MSE) 
- Backward pass: computing gradients with `loss.backward()`
- Parameter updates with optimizer (`torch.optim.SGD`)
- Zero gradients with `optimizer.zero_grad()`

### 3. Device-Agnostic Programming
- Checking for CUDA availability
- Moving data and models between CPU and GPU
- Using `to(device)` for tensor and model placement

### 4. Model Persistence
- Saving model state dict with `torch.save()`
- Loading models with `torch.load()`
- Best practices for model checkpointing

## Code Implementation: `linear_regression_complete.py`

The complete implementation demonstrates:
- **Data Generation**: Creating synthetic data with known parameters (weight=0.7, bias=0.3)
- **Model Definition**: `LinearRegressionModelV2` class with proper initialization
- **Training**: 1000 epochs with learning rate scheduling
- **Evaluation**: Model testing and comparison with original parameters
- **Visualization**: Plotting data, predictions, and training progress

## Key Takeaways
1. PyTorch's automatic differentiation makes gradient computation seamless
2. Device-agnostic code ensures compatibility across different hardware
3. Proper model saving/loading enables reproducible experiments
4. Visualization is crucial for understanding model behavior and training progress

## Next Steps
This foundation in linear regression prepares you for more complex models and classification tasks in subsequent lessons.