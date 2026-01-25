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
1. Install the package: pip install trainyourfly
2. Connectome data will be downloaded automatically on first run (~1.3GB)
3. Prepare your images in: {data_dir}/train/{class_name}/*.png

Example directory structure:
    my_project/
    ├── config.yaml
    ├── connectome_data/      # Downloaded automatically
    └── data/
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

Usage:
------
    # With default config
    python examples/simple_training.py

    # With YAML config
    python examples/simple_training.py --config config.yaml
"""

import os
import sys
import random
import argparse
from typing import List

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

# Ensure project root is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainyourfly.config import Config
from trainyourfly.utils.utils import get_image_paths


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


def train(config: Config):
    """
    Main training function.
    
    This demonstrates the complete training pipeline using the Drosophila
    connectome as the neural network architecture.
    """
    # Set random seeds for reproducibility
    if config.random_seed is not None:
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)
    
    if not os.path.exists(config.TRAINING_DATA_DIR):
        raise FileNotFoundError(
            f"Training data directory not found: {config.TRAINING_DATA_DIR}\n"
            f"Please create the directory structure:\n"
            f"  {config.data_dir}/train/class_name/*.png"
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
    if config.random_seed is not None:
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
    return model, data_processor


def evaluate(model, data_processor, config: Config):
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


def main():
    parser = argparse.ArgumentParser(
        description="Train a Drosophila connectome-based neural network"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default="data",
        help="Path to your dataset folder (must contain train/ and test/)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation on test set after training"
    )
    args = parser.parse_args()
    
    # Create configuration
    if args.config:
        print(f"Loading config from: {args.config}")
        config = Config.from_yaml(args.config)
    else:
        config = Config(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            NUM_CONNECTOME_PASSES=3,
        )
    
    # Train the model
    model, data_processor = train(config)
    
    # Optionally evaluate on test set
    if args.evaluate:
        evaluate(model, data_processor, config)


if __name__ == "__main__":
    main()
