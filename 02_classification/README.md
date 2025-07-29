# Lesson 02: Neural Network Classification with PyTorch

## Overview
This lesson explores binary and multi-class classification using neural networks, demonstrating the evolution from linear models to deep networks with non-linear activation functions.

## What You'll Learn
- Binary classification with circular/non-linear data
- The importance of non-linear activation functions
- Building multi-layer neural networks
- Classification metrics and evaluation
- Decision boundary visualization
- Multi-class and multi-label classification

## Key Concepts Covered

### 1. Classification vs Regression
- Output differences: continuous values vs discrete classes
- Loss functions: MSE vs CrossEntropy
- Activation functions for classification: Sigmoid, Softmax
- Evaluation metrics: Accuracy, Precision, Recall

### 2. Model Architecture Evolution
- **Linear Model**: Single layer, struggles with non-linear data
- **Multi-layer Model**: Multiple linear layers, still linear overall
- **Non-linear Model**: Adding ReLU activation functions for non-linearity

### 3. Activation Functions
- **ReLU (Rectified Linear Unit)**: `max(0, x)` - introduces non-linearity
- **Sigmoid**: Outputs between 0 and 1, good for binary classification
- **Tanh**: Outputs between -1 and 1, zero-centered

### 4. Loss Functions for Classification
- **Binary Cross-Entropy**: For binary classification problems
- **Cross-Entropy**: For multi-class classification
- **BCE with Logits**: Combines sigmoid and BCE for numerical stability

## Code Implementations

### Primary: `binary_classification_complete.py`
The complete binary classification implementation featuring:
- **CircleModelV2**: 3-layer network with ReLU activation
- **Non-linear Data**: Circular pattern that requires non-linearity to solve
- **Training Loop**: Complete with loss tracking and evaluation
- **Visualization**: Decision boundary plotting to understand model behavior

### Additional Files
- `02_pytorch_classification_video.py`: Multi-class classification with blob data
- `multilabel_dataset.py`: Multi-label classification example
- `02_pytorch_classification_video.ipynb`: Interactive Jupyter notebook version

## Key Insights

### The Non-linearity Problem
- Linear models can only learn linear decision boundaries
- Non-linear data (like circles, spirals) requires non-linear activation functions
- ReLU activation enables the network to learn complex patterns

### Decision Boundaries
- Visualization reveals how the model separates different classes
- Linear models create straight decision boundaries
- Non-linear models can create curved, complex boundaries

### Training Considerations
- Learning rate affects convergence speed and stability
- More complex models may require more training epochs
- Regularization becomes important with increased model complexity

## Model Progression
1. **CircleModelV0**: 2 layers, linear - fails on circular data
2. **CircleModelV1**: 3 layers, still linear - marginal improvement
3. **CircleModelV2**: 3 layers with ReLU - successfully learns circular patterns

## Key Takeaways
1. Non-linear activation functions are essential for complex pattern recognition
2. Model architecture significantly impacts learning capability
3. Visualization helps understand model behavior and decision-making
4. Different problems require different classification approaches (binary vs multi-class)

## Next Steps
This classification foundation prepares you for computer vision tasks where spatial patterns and feature hierarchies become crucial.