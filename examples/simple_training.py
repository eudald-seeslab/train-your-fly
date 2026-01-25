"""
Simple Training Example for TrainYourFly
=========================================

This script demonstrates how to use the trainyourfly library to train a 
Drosophila connectome-based neural network on visual classification tasks.

The model processes images through:
1. Voronoi tessellation (mimicking the fly's compound eye)
2. Neuron activation mapping (R1-6, R7, R8 photoreceptors)
3. Message passing through the connectome graph
4. Decision-making layer for classification

Prerequisites:
-------------
1. Install the package: pip install -e .
2. Download connectome data from GitHub releases and extract to the project root
   See: https://github.com/ecorreig/train-your-fly/releases
3. Prepare your images in: images/{task_name}/train/{class_name}/*.png

Example directory structure:
    images/
    └── my_task/
        ├── train/
        │   ├── class_a/
        │   │   ├── img001.png
        │   │   └── img002.png
        │   └── class_b/
        │       ├── img001.png
        │       └── img002.png
        └── test/
            ├── class_a/
            │   └── ...
            └── class_b/
                └── ...
"""

import os
import sys
import random
from dataclasses import dataclass, field
from typing import List, Optional, Callable

import numpy as np
import torch
from torch import nn
from torch.nn.functional import leaky_relu
from torch.optim import AdamW
from tqdm import tqdm

# Ensure project root is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import PROJECT_ROOT


def check_data_exists():
    """Check if required connectome data files exist."""
    data_dir = os.path.join(PROJECT_ROOT, "connectome_data")
    required_files = [
        "classification.csv",
        "right_visual_positions_all_neurons.csv",
    ]
    
    missing = []
    if not os.path.exists(data_dir):
        missing.append(f"Directory '{data_dir}' does not exist")
    else:
        for f in required_files:
            if not os.path.exists(os.path.join(data_dir, f)):
                missing.append(f)
    
    if missing:
        raise FileNotFoundError(
            f"Missing connectome data files: {missing}\n\n"
            "Please download the connectome data from the GitHub releases:\n"
            "  https://github.com/ecorreig/train-your-fly/releases\n\n"
            "Extract the zip file to the project root."
        )


@dataclass
class SimpleConfig:
    """
    Minimal configuration for training.
    
    This dataclass contains all the parameters needed to configure the model.
    For a full list of options, see configs/config.py
    """
    # Data paths
    data_type: str = "my_task"  # Name of your task folder in images/
    CONNECTOME_DATA_DIR: str = "connectome_data"
    
    # Training parameters
    batch_size: int = 4
    num_epochs: int = 10
    base_lr: float = 0.0003
    
    # Model architecture
    NUM_CONNECTOME_PASSES: int = 3  # How many times to propagate through the graph
    train_edges: bool = True  # Train synaptic weights
    train_neurons: bool = False  # Train neuron activation thresholds
    final_layer: str = "mean"  # "mean" or "nn" - how to aggregate decision neurons
    
    # Biological parameters
    eye: str = "right"  # Which eye to use ("left" or "right")
    neurons: str = "all"  # "all" or "selected" neurons
    voronoi_criteria: str = "R7"  # Tessellation based on R7 neurons
    inhibitory_r7_r8: bool = False  # R7/R8 mutual inhibition
    rational_cell_types: List[str] = field(
        default_factory=lambda: ["KCapbp-m", "KCapbp-ap2", "KCapbp-ap1"]
    )
    num_decision_making_neurons: Optional[int] = None
    
    # Normalization and regularization
    neuron_normalization: str = "min_max"  # "min_max" or "log1p"
    lambda_func: Callable = leaky_relu
    neuron_dropout: float = 0.0
    decision_dropout: float = 0.0
    filtered_celltypes: List[str] = field(default_factory=list)
    filtered_fraction: float = 0.25
    
    # Synaptic constraints
    refined_synaptic_data: bool = False
    synaptic_limit: bool = True
    log_transform_weights: bool = False
    new_connectome: bool = True
    randomization_strategy: Optional[str] = None  # None, "unconstrained", "binned", etc.
    
    # Device and precision
    device_type: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    sparse_layout: torch.layout = torch.sparse_coo
    random_seed: int = 42
    
    # Computed properties (set in __post_init__)
    DEVICE: torch.device = field(init=False)
    TRAINING_DATA_DIR: str = field(init=False)
    TESTING_DATA_DIR: str = field(init=False)
    CLASSES: List[str] = field(init=False)
    
    def __post_init__(self):
        self.DEVICE = torch.device(self.device_type)
        self.TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "images", self.data_type, "train")
        self.TESTING_DATA_DIR = os.path.join(PROJECT_ROOT, "images", self.data_type, "test")
        
        if os.path.exists(self.TRAINING_DATA_DIR):
            self.CLASSES = sorted(os.listdir(self.TRAINING_DATA_DIR))
        else:
            self.CLASSES = []


def get_image_paths(directory: str, limit: Optional[int] = None) -> List[str]:
    """Get all image paths from a directory with class subdirectories."""
    paths = []
    for class_name in sorted(os.listdir(directory)):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    paths.append(os.path.join(class_dir, img_name))
    
    if limit:
        paths = paths[:limit]
    return paths


def select_random_batch(
    all_paths: List[str], 
    batch_size: int, 
    already_used: List[str]
) -> tuple:
    """Select a random batch of images, avoiding already-used ones."""
    available = [p for p in all_paths if p not in already_used]
    
    if len(available) < batch_size:
        # Reset if we've used most images
        available = all_paths
        already_used = []
    
    selected = random.sample(available, min(batch_size, len(available)))
    already_used.extend(selected)
    
    return selected, already_used


def train(config: SimpleConfig):
    """
    Main training function.
    
    This demonstrates the complete training pipeline using the Drosophila
    connectome as the neural network architecture.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    
    # Check data exists
    check_data_exists()
    
    if not os.path.exists(config.TRAINING_DATA_DIR):
        raise FileNotFoundError(
            f"Training data directory not found: {config.TRAINING_DATA_DIR}\n"
            f"Please create the directory structure:\n"
            f"  images/{config.data_type}/train/class_name/*.png"
        )
    
    print(f"Training on device: {config.DEVICE}")
    print(f"Classes: {config.CLASSES}")
    
    # Import here to avoid import errors if data is missing
    from trainyourfly.data.data_processing import DataProcessor
    from trainyourfly.connectome_models.graph_models import FullGraphModel
    
    # Initialize data processor (loads connectome, sets up Voronoi tessellation)
    print("Initializing data processor (this may take a moment)...")
    data_processor = DataProcessor(config)
    
    # Create the connectome-based model
    random_generator = torch.Generator(device=config.DEVICE)
    random_generator.manual_seed(config.random_seed)
    
    model = FullGraphModel(data_processor, config, random_generator).to(config.DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Setup training
    optimizer = AdamW(model.parameters(), lr=config.base_lr)
    criterion = nn.CrossEntropyLoss()
    
    # Get training images
    training_images = get_image_paths(config.TRAINING_DATA_DIR)
    print(f"Found {len(training_images)} training images")
    
    iterations_per_epoch = len(training_images) // config.batch_size
    
    # Training loop
    model.train()
    for epoch in range(config.num_epochs):
        already_selected = []
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(range(iterations_per_epoch), desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for i in pbar:
            # Get batch
            batch_paths, already_selected = select_random_batch(
                training_images, config.batch_size, already_selected
            )
            
            # Load and process images
            images, labels = data_processor.get_data_from_paths(batch_paths)
            inputs, labels = data_processor.process_batch(images, labels)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/(i+1):.4f}',
                'acc': f'{100.*correct/total:.1f}%'
            })
        
        print(f"Epoch {epoch+1}: Loss={running_loss/iterations_per_epoch:.4f}, "
              f"Accuracy={100.*correct/total:.1f}%")
    
    print("\nTraining complete!")
    return model


def evaluate(model, data_processor, config: SimpleConfig):
    """Evaluate the model on test data."""
    if not os.path.exists(config.TESTING_DATA_DIR):
        print(f"No test directory found at {config.TESTING_DATA_DIR}")
        return
    
    test_images = get_image_paths(config.TESTING_DATA_DIR)
    if not test_images:
        print("No test images found")
        return
    
    print(f"\nEvaluating on {len(test_images)} test images...")
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        already_selected = []
        iterations = len(test_images) // config.batch_size
        
        for _ in tqdm(range(iterations), desc="Evaluating"):
            batch_paths, already_selected = select_random_batch(
                test_images, config.batch_size, already_selected
            )
            
            images, labels = data_processor.get_data_from_paths(batch_paths)
            inputs, labels = data_processor.process_batch(images, labels)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    print(f"Test Accuracy: {100.*correct/total:.1f}%")


if __name__ == "__main__":
    # Create configuration
    config = SimpleConfig(
        data_type="my_task",  # Change this to your task folder name
        batch_size=4,
        num_epochs=5,
        NUM_CONNECTOME_PASSES=3,
    )
    
    # Train the model
    model = train(config)
    
    # Optionally evaluate on test set
    # from trainyourfly.data.data_processing import DataProcessor
    # data_processor = DataProcessor(config)
    # evaluate(model, data_processor, config)
