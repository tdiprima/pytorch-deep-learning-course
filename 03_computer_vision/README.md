# Lesson 03: Computer Vision with Convolutional Neural Networks

## Overview
This lesson introduces computer vision fundamentals using Convolutional Neural Networks (CNNs) on the Fashion-MNIST dataset, progressing from basic neural networks to specialized vision architectures.

## What You'll Learn
- Computer vision dataset handling and preprocessing
- Convolutional Neural Network (CNN) architecture
- Fashion-MNIST classification
- Model evaluation with confusion matrices
- Training and testing pipeline optimization
- GPU acceleration for computer vision

## Key Concepts Covered

### 1. Computer Vision Fundamentals
- **Image Data**: Understanding tensors for images (height, width, channels)
- **Fashion-MNIST**: 28x28 grayscale images of clothing items (10 classes)
- **Data Loading**: Using `torchvision.datasets` and `DataLoader`
- **Normalization**: Preprocessing images for better training

### 2. Convolutional Neural Networks (CNNs)
- **Convolutional Layers**: Feature extraction through convolution operations
- **Pooling Layers**: Spatial dimension reduction (MaxPool2d)
- **Flattening**: Converting 2D feature maps to 1D for classification
- **Feature Hierarchies**: Learning from edges to complex patterns

### 3. Model Architecture Comparison
- **Baseline**: Simple linear model (flattened pixels → classes)
- **Non-linear**: Multi-layer perceptron with ReLU activation
- **CNN**: Convolutional architecture optimized for spatial data

### 4. Advanced Training Techniques
- **Progress Tracking**: Using tqdm for training progress bars
- **Timing**: Measuring training and inference time
- **Device Management**: Efficient GPU utilization
- **Modular Code**: Separating training and testing functions

## Code Implementation: `fashion_mnist_cnn_complete.py`

The complete implementation features:

### Model Architecture (FashionMNISTModelV2)
```python
- Conv2d(1, 10, 3) → ReLU → Conv2d(10, 10, 3) → ReLU → MaxPool2d(2)
- Conv2d(10, 10, 3) → ReLU → Conv2d(10, 10, 3) → ReLU → MaxPool2d(2)
- Flatten → Linear(250, 10)
```

### Key Functions
- `train_step()`: Single training epoch with loss tracking
- `test_step()`: Model evaluation with accuracy calculation
- `eval_model()`: Comprehensive model assessment
- Modular design for reusability across projects

### Training Pipeline
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: CrossEntropyLoss for multi-class classification
- **Epochs**: Configurable training duration
- **Evaluation**: Accuracy tracking and confusion matrix generation

## Key Insights

### Why CNNs Excel at Computer Vision
1. **Spatial Awareness**: Convolution preserves spatial relationships
2. **Parameter Sharing**: Same filters applied across the image
3. **Translation Invariance**: Features detected regardless of position
4. **Hierarchical Learning**: Simple to complex feature progression

### Performance Comparison
- **Linear Model**: ~85% accuracy (treats pixels independently)
- **Non-linear MLP**: ~87% accuracy (better feature combinations)
- **CNN**: ~90%+ accuracy (spatial feature extraction)

### Training Optimization
- **GPU Acceleration**: Significant speedup for image processing
- **Batch Processing**: Efficient parallel computation
- **Progress Monitoring**: Real-time training feedback

## Dataset Details: Fashion-MNIST
- **Classes**: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- **Size**: 60,000 training + 10,000 test images
- **Format**: 28×28 grayscale images
- **Challenge**: More complex than MNIST digits, closer to real-world vision tasks

## Key Takeaways
1. CNNs are specifically designed for spatial data like images
2. Convolutional layers extract meaningful visual features automatically
3. Proper data preprocessing and normalization are crucial
4. Modular code design enables experimentation and reusability
5. GPU acceleration is essential for efficient computer vision training

## Files in Directory
- `fashion_mnist_cnn_complete.py`: Complete CNN implementation
- `computer_vision_gpu.ipynb`: Interactive notebook with GPU examples
- `display_images.ipynb`: Visualization and data exploration notebook

## Next Steps
This computer vision foundation prepares you for custom datasets and more advanced architectures like ResNet, transfer learning, and object detection.