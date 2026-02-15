# Train Your Fly

A library for training biologically constrained neural networks using the *Drosophila melanogaster* connectome.

This library processes visual input through a model of the fly's compound eye and propagates signals through the actual synaptic connectivity of the fly brain.

**Note:** this repo was integrated with [this one](https://github.com/eudald-seeslab/connectome) as a part of a research project. I am in the process of decoupling it completely, but there might be some leftover code that makes little sense as it is. Furthremore, this is also why you will see a milion configuration parameters in config.yaml, a fair amount of which are quite useless; I will clean it at some point.

## Installation

```bash
git clone https://github.com/ecorreig/train-your-fly.git
cd train-your-fly
pip install -e .
```

## Quick Start

```python
from trainyourfly import Config, train, evaluate

config = Config(data_dir="my_data", num_epochs=10)
result = train(config)
accuracy = evaluate(result)
```

That's it. `result` contains the trained model, data processor, and training history.

You can customise the optimizer, loss, or plug in experiment tracking:

```python
from torch import nn

result = train(
    config,
    criterion=nn.CrossEntropyLoss(label_smoothing=0.1),
    tracker=my_tracker,  # WandBTracker, CSVTracker, etc.
)
```

See `quickstart.ipynb` for an interactive tutorial or `examples/` for complete scripts.

## Data Structure

There are two types of data needed, the connectome data, and the images used for training and testing. The two need to follow this structure:

### Connectome data

The connectome data (~1.3GB) is downloaded automatically on first run. You can also download it manually from the [releases page](https://github.com/ecorreig/train-your-fly/releases/latest).

The connectome data is derived from [FlyWire](https://flywire.ai/). Please cite the original work when using this data.

It resides in the connectome_data directory.

### Train/test images

The directory schema is:
```
images/       
├── train/
   ├── class_a/
   │   ├── img001.png
   │   └── img002.png
   └── class_b/
       └── ...
└── test/
   ├── class_a/
   │   └── ...
   └── class_b/
        └── ...
```

I created the images using the [CogStim](https://github.com/eudald-seeslab/cogstim). For example, for approximate number system images, you may use:

```bash
cogstim -ans --train-num 100 --test-num 40
```


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

## Experiment Tracking

The library is agnostic to the experiment tracking tool you use. An `ExperimentTracker` protocol defines the interface that any tracker must satisfy:

```python
class ExperimentTracker(Protocol):
    def initialize(self, config) -> None: ...
    def log_metrics(self, epoch, loss, accuracy, *, task=None) -> None: ...
    def log_image(self, figure, name, title, *, task=None) -> None: ...
    def log_dataframe(self, df, title) -> None: ...
    def log_validation(self, loss, accuracy, results_df, plots, *, task=None) -> None: ...
    def finish(self) -> None: ...
```

Pass any tracker that implements these methods to the training function. By default a `NullTracker` is used (no tracking).

### Using Weights & Biases

Install the optional dependency and use the built-in `WandBTracker`:

```bash
pip install train-your-fly[wandb]
# or: pip install wandb
```

```python
from trainyourfly.integrations.wandb_tracker import WandBTracker

tracker = WandBTracker(project="my-fly-project", group="experiment-1")
train(config, tracker=tracker)
```

See `examples/training_with_wandb.py` for a full script.

### Writing Your Own Tracker

You can integrate any tracking tool (MLflow, TensorBoard, CSV files, ...) by implementing the same methods. See `examples/training_with_csv_logger.py` for a complete example that writes metrics to a CSV file and saves plots to disk:

```python
class CSVTracker:
    def initialize(self, config):
        self._file = open("metrics.csv", "w")
        # ...

    def log_metrics(self, epoch, loss, accuracy, *, task=None):
        self._file.write(f"{epoch},{loss},{accuracy}\n")

    def log_image(self, figure, name, title, *, task=None):
        figure.savefig(f"images/{title}_{name}.png")

    def finish(self):
        self._file.close()
```

## Logging

The library uses Python's standard `logging` module for console output. All messages go through the `trainyourfly` logger, which is configured with coloured formatting by default. You can control verbosity:

```python
import logging

# Quieter (only warnings and errors)
logging.getLogger("trainyourfly").setLevel(logging.WARNING)

# More verbose (includes debug messages)
logging.getLogger("trainyourfly").setLevel(logging.DEBUG)
```

## License

MIT License - see [LICENSE](LICENSE) for details.
