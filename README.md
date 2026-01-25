# Train Your Fly

A library for training biologically constrained neural networks using the *Drosophila melanogaster* connectome.

This library processes visual input through a model of the fly's compound eye and propagates signals through the actual synaptic connectivity of the fly brain.

## Installation

```bash
git clone https://github.com/ecorreig/train-your-fly.git
cd train-your-fly
pip install -e .
```

## Quick Start

```python
from trainyourfly.config import Config
from trainyourfly.data.data_processing import DataProcessor
from trainyourfly.connectome_models.graph_models import FullGraphModel
from trainyourfly.utils.utils import get_image_paths

# Create configuration
config = Config(
    data_dir="data",       # Path to your data folder
    batch_size=8,
)

# Initialize data processor (downloads connectome automatically if needed)
data_processor = DataProcessor(config)

# Create the connectome-based model
model = FullGraphModel(data_processor, config).to(config.DEVICE)

# Process a batch of images
train_images = get_image_paths(config.TRAINING_DATA_DIR)[:config.batch_size]
images, labels = data_processor.get_data_from_paths(train_images)
inputs, labels = data_processor.process_batch(images, labels)

# Forward pass through the connectome
outputs = model(inputs)
```

See `quickstart.ipynb` for an interactive tutorial or `examples/simple_training.py` for a complete training script.

## Data Structure

Your data folder must have this structure:

```
my_project/
├── config.yaml           # Optional: your configuration
├── connectome_data/      # Downloaded automatically (~1.3GB)
└── data/                 # Your images (set via data_dir)
    ├── train/
    │   ├── class_a/
    │   │   ├── img001.png
    │   │   └── img002.png
    │   └── class_b/
    │       └── ...
    └── test/
        ├── class_a/
        │   └── ...
        └── class_b/
            └── ...
```

The connectome data (~1.3GB) is downloaded automatically on first run. You can also download it manually from the [releases page](https://github.com/ecorreig/train-your-fly/releases/latest).

The connectome data is derived from [FlyWire](https://flywire.ai/). Please cite the original work when using this data.

## How It Works

1. **Voronoi Tessellation**: Images are divided into regions corresponding to the fly's ommatidia (eye units)
2. **Photoreceptor Activation**: Each region activates R1-6, R7, and R8 photoreceptors based on color
3. **Connectome Propagation**: Signals propagate through the actual synaptic connections of the fly brain
4. **Decision Layer**: Output neurons (mushroom body) are used for classification

## Configuration

The easiest way to configure your experiments is with a YAML file:

```python
from trainyourfly import Config

# Create an example config.yaml with all parameters documented
Config.create_example("config.yaml")

# Edit the file, then load it
config = Config.from_yaml("config.yaml")
```

You can also create a config directly in Python:

```python
config = Config(data_dir="my_data", batch_size=16, num_epochs=50)

# Save your config for reproducibility
config.to_yaml("my_experiment.yaml")
```

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `data_dir` | Path to your data folder (with train/ and test/ subfolders) |
| `connectome_data_dir` | Path to connectome data (default: `"connectome_data"`) |
| `NUM_CONNECTOME_PASSES` | Number of message-passing iterations through the graph |
| `train_edges` | Whether to train synaptic weights |
| `train_neurons` | Whether to train neuron activation thresholds |
| `eye` | Which eye to use (`"left"` or `"right"`) |
| `voronoi_criteria` | Tessellation method (`"R7"` recommended) |
| `rational_cell_types` | Neuron types used for decision-making |
| `filtered_fraction` | Fraction of neurons to ablate (for experiments) |

See the generated `config.yaml` for the full list of parameters with descriptions.

## License

MIT License - see [LICENSE](LICENSE) for details.
