# Train Your Fly

A library for training biologically constrained neural networks using the *Drosophila melanogaster* connectome.

This library processes visual input through a model of the fly's compound eye and propagates signals through the actual synaptic connectivity of the fly brain.

## Installation

```bash
git clone https://github.com/ecorreig/train-your-fly.git
cd train-your-fly
pip install -e .
```

### Download Connectome Data

The connectome data is required but not included in the repository due to its size.

1. Download `data.zip` from the [latest release](https://github.com/ecorreig/train-your-fly/releases/latest)
2. Extract it to the project root:
   ```bash
   unzip data.zip -d /path/to/train-your-fly/
   ```

The connectome data is derived from [FlyWire](https://flywire.ai/). Please cite the original work when using this data.

## Quick Start

```python
from trainyourfly.data.data_processing import DataProcessor
from trainyourfly.connectome_models.graph_models import FullGraphModel
from configs import config

TRAIN_IMAGES_DIR = "images/ans/train"
train_images = get_image_paths(TRAIN_IMAGES_DIR)

# Initialize the data processor (loads connectome, sets up Voronoi tessellation)
data_processor = DataProcessor(config)

# Create the connectome-based model
model = FullGraphModel(data_processor, config).to(config.DEVICE)

# Process a batch of images
images, labels = data_processor.get_data_from_paths(train_images)
inputs, labels = data_processor.process_batch(images, labels)

# Forward pass through the connectome
outputs = model(inputs)
```

See `examples/simple_training.py` for a complete training example.

## How It Works

1. **Voronoi Tessellation**: Images are divided into regions corresponding to the fly's ommatidia (eye units)
2. **Photoreceptor Activation**: Each region activates R1-6, R7, and R8 photoreceptors based on color
3. **Connectome Propagation**: Signals propagate through the actual synaptic connections of the fly brain
4. **Decision Layer**: Output neurons (mushroom body) are used for classification

## Project Structure

```
train-your-fly/
├── configs/           # Configuration files
├── connectome_data/   # Connectome data files (download from releases)
├── examples/          # Example scripts
├── src/trainyourfly/
│   ├── connectome_models/  # Graph neural network models
│   ├── data/              # Data processing pipeline
│   ├── eye_models/        # Voronoi and neuron mapping
│   ├── plots/             # Visualization utilities
│   └── utils/             # Helper functions
└── tests/             # Unit tests
```

## Configuration

Key parameters in `configs/config.py`:

| Parameter | Description |
|-----------|-------------|
| `NUM_CONNECTOME_PASSES` | Number of message-passing iterations through the graph |
| `train_edges` | Whether to train synaptic weights |
| `train_neurons` | Whether to train neuron activation thresholds |
| `eye` | Which eye to use (`"left"` or `"right"`) |
| `voronoi_criteria` | Tessellation method (`"R7"` recommended) |
| `rational_cell_types` | Neuron types used for decision-making |

## License

MIT License - see [LICENSE](LICENSE) for details.
