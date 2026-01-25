"""
Configuration management for TrainYourFly.

This module provides a dataclass-based configuration system that can be
instantiated directly with keyword arguments or loaded from a YAML file.

Example usage:

    # Direct instantiation with defaults
    config = Config()

    # Override specific parameters
    config = Config(batch_size=16, num_epochs=50)

    # Load from YAML file
    config = Config.from_yaml("my_config.yaml")

    # Save current config to YAML
    config.to_yaml("saved_config.yaml")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Callable, Any

import torch
from torch.nn.functional import leaky_relu, relu, tanh, sigmoid


# Mapping from string names to activation functions
ACTIVATION_FUNCTIONS = {
    "leaky_relu": leaky_relu,
    "relu": relu,
    "tanh": tanh,
    "sigmoid": sigmoid,
}

# Mapping from string names to torch dtypes
TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}

# Mapping from string names to sparse layouts
SPARSE_LAYOUTS = {
    "sparse_coo": torch.sparse_coo,
    "sparse_csr": torch.sparse_csr,
}


@dataclass
class Config:
    """Configuration for TrainYourFly models and training.
    
    This dataclass contains all parameters needed to configure the connectome
    model, training process, and data handling. Parameters have sensible defaults
    for most use cases.
    
    Attributes are grouped into categories:
    - Data: paths and dataset configuration
    - Training: optimization and training loop parameters
    - Model: architecture and connectome parameters
    - Biological: neuroscience-inspired parameters
    - Debugging: debugging options
    - Visualization: plotting configuration
    """
    
    # =========================================================================
    # Data Configuration
    # =========================================================================
    
    data_dir: str = "data"
    """Path to your dataset folder (relative to current directory or absolute).
    
    The folder must contain exactly two subfolders:
        {data_dir}/train/{class_name}/*.png
        {data_dir}/test/{class_name}/*.png
    """
    
    connectome_data_dir: str = "connectome_data"
    """Directory containing connectome CSV files.
    
    If the directory doesn't exist, you will be prompted to download
    the data (~1.3GB) from GitHub releases.
    """
    
    # =========================================================================
    # Training Configuration
    # =========================================================================
    
    batch_size: int = 8
    """Number of images per training batch."""
    
    num_epochs: int = 100
    """Number of training epochs."""
    
    base_lr: float = 0.0003
    """Base learning rate for optimizer."""
    
    patience: int = 2
    """Early stopping patience (epochs without improvement)."""
    
    save_every_checkpoint: bool = False
    """Whether to save model after every epoch."""
    
    resume_checkpoint: Optional[str] = None
    """Path to checkpoint file to resume training from."""
    
    random_seed: Optional[int] = 42
    """Random seed for reproducibility. Set to None for random initialization."""
    
    # =========================================================================
    # Model Architecture
    # =========================================================================
    
    NUM_CONNECTOME_PASSES: int = 3
    """Number of message passing iterations through the connectome graph."""
    
    train_edges: bool = True
    """Whether to train (learn) synaptic edge weights."""
    
    train_neurons: bool = False
    """Whether to train neuron activation thresholds."""
    
    neurons: str = "all"
    """Which neurons to include: 'all' or 'selected'."""
    
    final_layer: str = "mean"
    """Final aggregation method: 'mean' or 'nn' (neural network layer)."""
    
    num_decision_making_neurons: Optional[int] = None
    """Number of neurons for final decision. None uses all decision neurons."""
    
    activation_function: str = "leaky_relu"
    """Activation function name: 'leaky_relu', 'relu', 'tanh', 'sigmoid'."""
    
    neuron_normalization: str = "min_max"
    """Normalization method: 'min_max' or 'log1p'."""
    
    # =========================================================================
    # Biological Parameters
    # =========================================================================
    
    eye: str = "right"
    """Which eye to simulate: 'left' or 'right'."""
    
    voronoi_criteria: str = "R7"
    """Voronoi tessellation criteria: 'R7' or 'all'."""
    
    inhibitory_r7_r8: bool = False
    """Enable R7/R8 mutual inhibition (as in biological retina)."""
    
    rational_cell_types: List[str] = field(
        default_factory=lambda: ["KCapbp-m", "KCapbp-ap2", "KCapbp-ap1"]
    )
    """Cell types used for final decision output."""
    
    filtered_celltypes: List[str] = field(default_factory=list)
    """Cell types to exclude from the network."""
    
    filtered_fraction: Optional[float] = 0.25
    """Fraction of non-essential neurons to randomly remove (ablation study)."""
    
    neuron_dropout: float = 0.0
    """Dropout rate for neuron activations during training."""
    
    decision_dropout: float = 0.0
    """Dropout rate for decision layer during training."""
    
    # =========================================================================
    # Synaptic Data Options
    # =========================================================================
    
    refined_synaptic_data: bool = False
    """Use refined (higher quality) synaptic connection data."""
    
    synaptic_limit: bool = True
    """Apply biological limits to synaptic weights."""
    
    log_transform_weights: bool = False
    """Apply log transform to synaptic weights."""
    
    new_connectome: bool = True
    """Use the newer (October 2024) connectome version."""
    
    randomization_strategy: Optional[str] = None
    """Synapse randomization: None, 'unconstrained', 'pruned', 'binned', etc."""
    
    # =========================================================================
    # Device and Precision
    # =========================================================================
    
    device_type: str = "auto"
    """Device type: 'auto', 'cuda', 'cpu', 'mps'."""
    
    dtype_str: str = "float32"
    """Tensor dtype: 'float32', 'float16', 'float64', 'bfloat16'."""
    
    sparse_layout_str: str = "sparse_coo"
    """Sparse tensor layout: 'sparse_coo', 'sparse_csr'."""
    
    # =========================================================================
    # Logging and Debugging
    # =========================================================================
    
    debugging: bool = False
    """Enable debugging mode (shorter runs, more verbose output)."""
    
    debug_length: int = 2
    """Number of batches per epoch in debug mode."""
    
    small_length: Optional[int] = None
    """Limit dataset size for quick experiments. None uses full dataset."""
    
    validation_length: Optional[int] = 400
    """Number of validation samples. None uses full validation set."""
    
    # =========================================================================
    # Visualization
    # =========================================================================
    
    plot_types: Optional[List[str]] = field(default_factory=list)
    """Plot types to generate. Empty list auto-detects from classes."""
    
    voronoi_colour: str = "#ff9933"
    """Colour for Voronoi cell boundaries in plots."""
    
    voronoi_width: int = 1
    """Line width for Voronoi boundaries in plots."""
    
    # =========================================================================
    # Computed Properties (set in __post_init__)
    # =========================================================================
    
    DEVICE: torch.device = field(init=False)
    """PyTorch device object (computed from device_type)."""
    
    dtype: torch.dtype = field(init=False)
    """PyTorch dtype (computed from dtype_str)."""
    
    sparse_layout: torch.layout = field(init=False)
    """PyTorch sparse layout (computed from sparse_layout_str)."""
    
    lambda_func: Callable = field(init=False)
    """Activation function (computed from activation_function)."""
    
    TRAINING_DATA_DIR: str = field(init=False)
    """Full path to training data directory."""
    
    TESTING_DATA_DIR: str = field(init=False)
    """Full path to testing data directory."""
    
    CONNECTOME_DATA_DIR: str = field(init=False)
    """Full path to connectome data directory."""
    
    CLASSES: List[str] = field(init=False)
    """List of class names (subdirectories in training data)."""
    
    def __post_init__(self):
        """Compute derived values after initialization."""
        # Compute device
        if self.device_type == "auto":
            if torch.cuda.is_available():
                self.DEVICE = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.DEVICE = torch.device("mps")
            else:
                self.DEVICE = torch.device("cpu")
        else:
            self.DEVICE = torch.device(self.device_type)
        
        # Compute dtype and sparse layout
        self.dtype = TORCH_DTYPES.get(self.dtype_str, torch.float32)
        self.sparse_layout = SPARSE_LAYOUTS.get(self.sparse_layout_str, torch.sparse_coo)
        
        # Compute activation function
        self.lambda_func = ACTIVATION_FUNCTIONS.get(self.activation_function, leaky_relu)
        
        # Compute data paths (direct paths, no images/ prefix)
        self.TRAINING_DATA_DIR = os.path.join(self.data_dir, "train")
        self.TESTING_DATA_DIR = os.path.join(self.data_dir, "test")
        self.CONNECTOME_DATA_DIR = self.connectome_data_dir
        
        # Get classes from directory
        if os.path.exists(self.TRAINING_DATA_DIR):
            self.CLASSES = sorted([
                d for d in os.listdir(self.TRAINING_DATA_DIR)
                if os.path.isdir(os.path.join(self.TRAINING_DATA_DIR, d))
            ])
        else:
            self.CLASSES = []
        
        # Debugging adjustments
        if self.debugging:
            self.num_epochs = min(self.num_epochs, 1)
        
        if self.small_length is None:
            self.validation_length = None
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file.
        
        Args:
            path: Path to the YAML configuration file.
            
        Returns:
            Config instance with values from the YAML file.
            
        Example:
            config = Config.from_yaml("config.yaml")
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config files. "
                "Install it with: pip install pyyaml"
            )
        
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        if data is None:
            data = {}
        
        # Handle nested YAML structure by flattening
        flat_data = cls._flatten_yaml(data)
        
        return cls(**flat_data)
    
    @staticmethod
    def _flatten_yaml(data: dict) -> dict:
        """Flatten nested YAML structure into flat config dict."""
        flat = {}
        for key, value in data.items():
            if isinstance(value, dict):
                # Nested section - extract values
                for sub_key, sub_value in value.items():
                    flat[sub_key] = sub_value
            else:
                flat[key] = value
        return flat
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file.
        
        Args:
            path: Path where the YAML file will be written.
            
        Example:
            config.to_yaml("my_config.yaml")
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config files. "
                "Install it with: pip install pyyaml"
            )
        
        # Get serializable dict
        data = self._to_serializable_dict()
        
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def _to_serializable_dict(self) -> dict:
        """Convert config to a dict that can be serialized to YAML."""
        # Get all fields that were init=True
        data = {}
        for f in self.__dataclass_fields__.values():
            if f.init:
                value = getattr(self, f.name)
                # Skip None values for cleaner YAML
                if value is not None:
                    data[f.name] = value
        return data
    
    def __repr__(self) -> str:
        """Pretty string representation."""
        lines = ["Config("]
        for f in self.__dataclass_fields__.values():
            if f.init:
                value = getattr(self, f.name)
                lines.append(f"    {f.name}={value!r},")
        lines.append(")")
        return "\n".join(lines)
    
    @staticmethod
    def create_example(path: str = "config.yaml", overwrite: bool = False) -> str:
        """Create an example configuration file with all parameters documented.
        
        This creates a well-commented YAML file that you can modify for your
        experiments. The file includes all available parameters with their
        default values and explanations.
        
        Args:
            path: Where to create the config file (default: "config.yaml")
            overwrite: If False (default), raises error if file exists
            
        Returns:
            The path to the created file.
            
        Example:
            >>> from trainyourfly import Config
            >>> Config.create_example("my_config.yaml")
            'my_config.yaml'
        """
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(
                f"File '{path}' already exists. Use overwrite=True to replace it."
            )
        
        # The example config content with full documentation
        example_config = '''\
# =============================================================================
# TrainYourFly Configuration
# =============================================================================
# This file contains all configuration parameters for training connectome-based
# neural networks. Modify the values below for your experiments.
#
# Usage:
#   from trainyourfly import Config
#   config = Config.from_yaml("config.yaml")
# =============================================================================


# -----------------------------------------------------------------------------
# Data Configuration
# -----------------------------------------------------------------------------

# Path to your dataset folder (relative to current directory or absolute).
# The folder must contain exactly two subfolders:
#   {data_dir}/train/{class_name}/*.png
#   {data_dir}/test/{class_name}/*.png
data_dir: "data"

# Directory containing connectome CSV files.
# If missing, you will be prompted to download (~1.3GB) from GitHub releases.
connectome_data_dir: "connectome_data"


# -----------------------------------------------------------------------------
# Training Configuration
# -----------------------------------------------------------------------------

# Number of images per training batch.
# Reduce if you get out-of-memory errors.
batch_size: 8

# Total number of training epochs
num_epochs: 100

# Base learning rate for the AdamW optimizer
base_lr: 0.0003

# Early stopping patience: number of epochs without improvement before stopping
patience: 2

# Whether to save a checkpoint after every epoch
save_every_checkpoint: false

# Path to a checkpoint file to resume training from
# resume_checkpoint: null

# Random seed for reproducibility. Set to null for random initialization.
random_seed: 42


# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------

# Number of message passing iterations through the connectome graph.
# More passes allow signals to propagate deeper through the network.
NUM_CONNECTOME_PASSES: 3

# Whether to train (learn) the synaptic edge weights.
# This is the main source of trainable parameters.
train_edges: true

# Whether to train neuron activation thresholds.
train_neurons: false

# Which neurons to include in the model
neurons: "all"  # "all" or "selected"

# Method to aggregate decision neuron outputs for final classification
final_layer: "mean"  # "mean" or "nn"

# Limit the number of decision neurons (null = use all)
num_decision_making_neurons: null

# Activation function for neuron embeddings
activation_function: "leaky_relu"  # leaky_relu, relu, tanh, sigmoid

# Normalization method for connectome output
neuron_normalization: "min_max"  # min_max, log1p


# -----------------------------------------------------------------------------
# Biological Parameters
# -----------------------------------------------------------------------------

# Which eye to simulate
eye: "right"  # "left" or "right"

# Voronoi tessellation criteria
voronoi_criteria: "R7"  # "R7" or "all"

# Enable R7/R8 mutual inhibition (as in biological retina)
inhibitory_r7_r8: false

# Cell types used for final decision output (mushroom body Kenyon cells)
rational_cell_types:
  - "KCapbp-m"
  - "KCapbp-ap2"
  - "KCapbp-ap1"

# Cell types to exclude from the network (ablation studies)
filtered_celltypes: []

# Fraction of neurons to randomly remove (ablation studies, null = none)
filtered_fraction: null

# Dropout rates
neuron_dropout: 0.0
decision_dropout: 0.0


# -----------------------------------------------------------------------------
# Synaptic Data Options
# -----------------------------------------------------------------------------

# Use refined (higher quality) synaptic connection data
refined_synaptic_data: false

# Apply biological limits to synaptic weights
synaptic_limit: true

# Apply log transform to synaptic weights
log_transform_weights: false

# Use the newer (October 2024) connectome version
new_connectome: true

# Synapse randomization for control experiments
# Options: null, "unconstrained", "pruned", "conn_pruned", "binned", "neuron_binned"
randomization_strategy: null


# -----------------------------------------------------------------------------
# Device and Precision
# -----------------------------------------------------------------------------

# Device to use for computation
device_type: "auto"  # "auto", "cuda", "cpu", "mps"

# Tensor data type
dtype_str: "float32"  # float32, float16, float64, bfloat16

# Sparse tensor layout
sparse_layout_str: "sparse_coo"  # sparse_coo, sparse_csr


# -----------------------------------------------------------------------------
# Debugging
# -----------------------------------------------------------------------------

# Enable debugging mode (shorter runs, more verbose output)
debugging: false

# Number of batches per epoch when debugging
debug_length: 2

# Limit dataset size for quick experiments (null = full dataset)
small_length: null

# Number of validation samples (null = full validation set)
validation_length: 400


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

# Plot types to generate (empty = auto-detect from classes)
plot_types: []

# Voronoi visualization settings
voronoi_colour: "#ff9933"
voronoi_width: 1
'''
        
        with open(path, "w") as f:
            f.write(example_config)
        
        print(f"Created example config at '{path}'")
        print("Edit this file to configure your experiment, then load it with:")
        print(f"  config = Config.from_yaml('{path}')")
        
        return path
