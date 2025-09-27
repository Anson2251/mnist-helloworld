# Modular Neural Network Training Framework

A refactored, modular version of the original MNIST classifier that supports multiple datasets and model architectures.

## Features

### 🏗️ Modular Architecture
- **Dataset Abstraction**: Easy addition of new datasets
- **Model Registry**: Pluggable model architectures
- **Training Framework**: Reusable training components
- **Configuration Management**: YAML and command-line configuration

### 📊 Supported Datasets
- MNIST (28x28 grayscale, 10 classes)
- CIFAR-10 (32x32 RGB, 10 classes)
- Easy to extend with new datasets

### 🧠 Supported Models
- LeNet-5 (classic architecture)
- MyNet (custom architecture)
- Easy to add new architectures

## Project Structure

```
mnist-helloworld/
├── src/
│   ├── datasets/          # Dataset implementations
│   │   ├── base.py       # Base dataset class
│   │   ├── mnist.py      # MNIST dataset
│   │   ├── cifar.py      # CIFAR-10 dataset
│   │   └── registry.py   # Dataset registry
│   ├── models/           # Model implementations
│   │   ├── base.py       # Base model class
│   │   ├── lenet.py      # LeNet-5
│   │   ├── mynet.py      # Custom network
│   │   └── registry.py   # Model registry
│   ├── training/         # Training framework
│   │   ├── trainer.py    # Main trainer
│   │   ├── metrics.py    # Metrics tracking
│   │   └── checkpoint.py # Checkpoint management
│   ├── config/           # Configuration management
│   │   └── config.py     # Config parser and loader
│   └── utils/            # Utilities
│       ├── device.py     # Device detection
│       └── logger.py     # Logging setup
├── train.py              # Main training script
├── config.yaml           # Default configuration
└── requirements.txt      # Dependencies
```

## Usage

### Basic Usage

Train with default configuration (MNIST dataset, MyNet model):

```bash
python train.py
```

### Command Line Options

Train with CIFAR-10 dataset and LeNet model:

```bash
python train.py --dataset cifar10 --model lenet --epochs 30 --batch-size 128
```

Full list of options:

```bash
python train.py --help
```

### Configuration File

Use a YAML configuration file:

```bash
python train.py --config my_config.yaml
```

Example configuration:

```yaml
dataset:
  name: cifar10
  root: ./data
  download: true

model:
  name: lenet
  num_classes: 10

training:
  epochs: 30
  batch_size: 128
  num_workers: 8

optimization:
  learning_rate: 1e-3
  optimizer: adamw
```

## Adding New Components

### Adding a New Dataset

1. Create a new file in `src/datasets/`:

```python
from .base import BaseDataset
import torchvision.transforms as transforms

class MyDataset(BaseDataset):
    def get_train_transform(self):
        return transforms.Compose([...])
    
    def get_test_transform(self):
        return transforms.Compose([...])
    
    def load_data(self):
        # Load your dataset
        pass
    
    @property
    def num_classes(self):
        return 10
    
    @property
    def input_channels(self):
        return 3
    
    @property
    def input_size(self):
        return (32, 32)
```

2. Register it in `src/datasets/registry.py`:

```python
from .my_dataset import MyDataset
DatasetRegistry.register('mydataset', MyDataset)
```

### Adding a New Model

1. Create a new file in `src/models/`:

```python
from .base import BaseModel
import torch.nn as nn

class MyModel(BaseModel):
    def __init__(self, num_classes=10, input_channels=1, **kwargs):
        super().__init__(num_classes, input_channels)
        # Define your model architecture
        
    def forward(self, x):
        # Define forward pass
        return x
```

2. Register it in `src/models/registry.py`:

```python
from .my_model import MyModel
ModelRegistry.register('mymodel', MyModel)
```

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies:
- torch
- torchvision
- tqdm
- pyyaml

## Original Code

The original monolithic implementation is preserved in `lenet_mnist.py` for reference.