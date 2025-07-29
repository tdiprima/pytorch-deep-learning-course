# Lesson 04: Custom Datasets and TinyVGG Architecture

## Overview
This lesson teaches how to work with custom image datasets and implement the TinyVGG architecture, moving beyond standard datasets to real-world data scenarios with pizza, steak, and sushi images.

## What You'll Learn
- Creating custom PyTorch Dataset classes
- Image preprocessing and data augmentation
- TinyVGG architecture implementation
- Working with directory-structured datasets
- Data loading and transformation pipelines
- Multi-class classification on custom data

## Key Concepts Covered

### 1. Custom Dataset Creation
- **Dataset Class**: Inheriting from `torch.utils.data.Dataset`
- **Required Methods**: `__init__`, `__len__`, `__getitem__`
- **Data Organization**: Folder structure for different classes
- **Path Handling**: Using `pathlib` for robust file operations

### 2. Data Preprocessing Pipeline
- **Image Loading**: Using PIL for image handling
- **Transformations**: Resize, normalization, tensor conversion
- **Data Augmentation**: Random transformations to increase dataset diversity
- **Normalization**: StandardImageNet values for better training

### 3. TinyVGG Architecture
- **VGG-Style**: Stack of convolutional blocks
- **Conv Blocks**: Conv2d → ReLU → Conv2d → ReLU → MaxPool2d
- **Classification Head**: Adaptive pooling → flatten → linear classifier
- **Feature Maps**: Progressive increase in depth (3→10→10)

### 4. Directory Structure Pattern
```
data/
├── train/
│   ├── pizza/
│   ├── steak/
│   └── sushi/
└── test/
    ├── pizza/
    ├── steak/
    └── sushi/
```

## Code Implementation: `custom_dataset_tinyvgg_complete.py`

The complete implementation includes:

### TinyVGG Model Architecture
```python
- Conv Block 1: Conv2d(3,10,3) → ReLU → Conv2d(10,10,3) → ReLU → MaxPool2d(2)
- Conv Block 2: Conv2d(10,10,3) → ReLU → Conv2d(10,10,3) → ReLU → MaxPool2d(2)
- Classifier: AdaptiveAvgPool2d(1) → Flatten → Linear(10, num_classes)
```

### Key Components
- **Custom Dataset Class**: Handles image loading and labeling
- **Data Transforms**: Preprocessing pipeline for training and testing
- **Training Loop**: Complete training with loss and accuracy tracking
- **Model Evaluation**: Testing pipeline with performance metrics

### Training Configuration
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: CrossEntropyLoss for multi-class classification
- **Batch Size**: 32 (configurable)
- **Device**: Automatic GPU/CPU selection

## Key Insights

### Custom Dataset Advantages
1. **Real-world Data**: Work with your own images and classes
2. **Flexible Structure**: Adapt to any folder organization
3. **Custom Preprocessing**: Tailored transformations for your data
4. **Scalability**: Easy to add new classes or images

### TinyVGG Benefits
- **Lightweight**: Fewer parameters than full VGG networks
- **Effective**: Good performance on small to medium datasets
- **Educational**: Clear architecture that's easy to understand
- **Adaptable**: Can modify for different input sizes and classes

### Data Loading Best Practices
- **Consistent Transforms**: Same preprocessing for training and testing
- **Proper Normalization**: Use ImageNet stats for transfer learning compatibility
- **Batch Processing**: Efficient loading with DataLoader
- **Error Handling**: Robust file loading and processing

## Dataset Details: Pizza, Steak, Sushi
- **Classes**: 3 food categories with distinct visual characteristics
- **Challenge**: Real-world images with varying lighting, angles, backgrounds
- **Size**: Typically small custom datasets (few hundred images per class)
- **Preprocessing**: Resize to 64×64, normalize to ImageNet standards

## Architecture Details: TinyVGG

### Convolutional Blocks
- **Purpose**: Feature extraction through convolution and pooling
- **Activation**: ReLU for non-linearity
- **Pooling**: MaxPool2d reduces spatial dimensions

### Adaptive Pooling
- **Flexibility**: Handles variable input sizes
- **Efficiency**: Reduces parameters in classifier
- **Global Features**: Averages spatial information

## Key Takeaways
1. Custom datasets enable real-world problem solving
2. Proper data organization and loading are crucial for success
3. TinyVGG provides a good balance of performance and simplicity
4. Data preprocessing significantly impacts model performance
5. Custom Dataset classes provide flexibility for various data formats

## Removed Files (Development Stages)
- `custom_dataset_class.py`: Basic dataset class implementation
- `hello_dataset.py`: Dataset exploration and visualization
- `data_augmentation.py`: Data augmentation experiments
- `space_walk.py`: Directory structure utilities
- `display_rand_imgs.py`: Image visualization tools
- `download_data.py`: Data acquisition scripts
- `loss_curves.py`: Training visualization utilities

## Next Steps
This custom dataset foundation prepares you for transfer learning, where pre-trained models can be fine-tuned on your custom data for even better performance.