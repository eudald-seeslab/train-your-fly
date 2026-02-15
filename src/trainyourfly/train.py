"""
High-level training API for TrainYourFly.

This module exposes a ``train()`` function that handles the training
pipeline

Minimal usage::

    from trainyourfly import train, Config

    config = Config(data_dir="my_data")
    result = train(config)

With customisation::

    result = train(
        config,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        criterion=nn.CrossEntropyLoss(label_smoothing=0.1),
        tracker=WandBTracker(project="my-project"),
    )

Optimizer and criterion are configured here (not in ``Config``), so
there is a single source of truth.  Defaults are AdamW and
CrossEntropyLoss.

The function returns a ``TrainResult`` dataclass with the model, data
processor, and training history so you can inspect or continue training.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from os.path import basename
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from trainyourfly.config import Config
from trainyourfly.integrations.base import NullTracker
from trainyourfly.utils.utils import get_image_paths, select_random_images

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public result container
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Container returned by :func:`train`.

    Attributes
    ----------
    model : nn.Module
        The trained ``FullGraphModel``.
    data_processor : Any
        The ``DataProcessor`` instance (useful for evaluation / inference).
    history : dict
        Per-epoch metrics: ``{"loss": [...], "accuracy": [...]}``.
    config : Config
        The configuration that was used.
    """

    model: nn.Module
    data_processor: Any
    history: Dict[str, List[float]] = field(default_factory=lambda: {"loss": [], "accuracy": []})
    config: Config = None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train(
    config: Config,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    criterion: Optional[nn.Module] = None,
    tracker: Optional[Any] = None,
) -> TrainResult:
    """Train a connectome-based model from start to finish.

    Parameters
    ----------
    config : Config
        Full configuration (data paths, hyperparameters, etc.).
        Use ``config.debugging = True`` and ``config.debug_length`` to
        cap the number of batches per epoch for quick demos.
    optimizer : torch.optim.Optimizer, optional
        Optimizer instance.  When ``None`` (default), ``AdamW`` is used with
        ``lr=config.base_lr``.
    criterion : nn.Module, optional
        Loss function instance.  When ``None`` (default),
        ``CrossEntropyLoss`` is used.
    tracker : ExperimentTracker, optional
        Experiment tracker for metrics / images.  When ``None`` (default),
        a ``NullTracker`` is used (no tracking).

    Returns
    -------
    TrainResult
        Dataclass with ``model``, ``data_processor``, ``history``, and
        ``config``.

    Example
    -------
    ::

        from trainyourfly import Config, train

        # Simplest -- uses defaults (AdamW + CrossEntropyLoss)
        result = train(Config(data_dir="data", num_epochs=10))

        # Custom optimizer/criterion
        result = train(
            config,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            criterion=nn.CrossEntropyLoss(label_smoothing=0.1),
        )

        # With W&B tracking
        from trainyourfly.integrations.wandb_tracker import WandBTracker
        result = train(config, tracker=WandBTracker(project="my-project"))
    """
    # Lazy imports of heavy modules
    from trainyourfly.data.data_processing import DataProcessor
    from trainyourfly.connectome_models.graph_models import FullGraphModel

    # Reproducibility
    if config.random_seed is not None:
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

    # Initialise tracker
    tracker.initialize(config)

    # Data
    logger.info("Device: %s", config.DEVICE)
    logger.info("Loading connectome and building Voronoi tessellation...")
    data_processor = DataProcessor(config)
    logger.info("Classes: %s", data_processor.classes)
    logger.info("Neurons in connectome: %d", len(data_processor.root_ids))

    training_images = get_image_paths(config.TRAINING_DATA_DIR)
    logger.info("Training images: %d", len(training_images))

    # Model
    random_generator = torch.Generator(device=config.DEVICE)
    if config.random_seed is not None:
        random_generator.manual_seed(config.random_seed)

    model = FullGraphModel(data_processor, config, random_generator).to(config.DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{n_params:,}")

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.base_lr)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # Iterations per epoch
    iters = len(training_images) // config.batch_size
    if config.debugging:
        iters = min(iters, config.debug_length)

    # Training loop
    history: Dict[str, List[float]] = {"loss": [], "accuracy": []}

    model.train()
    for epoch in range(config.num_epochs):
        used: List[str] = []
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(range(iters), desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for i in pbar:
            batch_paths, used = select_random_images(
                training_images, config.batch_size, used
            )

            images, labels = data_processor.get_data_from_paths(batch_paths)

            # Log a Voronoi image at the start of each epoch
            if i == 0:
                fig, title = data_processor.plot_input_images(
                    images[0], config.voronoi_colour, config.voronoi_width
                )
                tracker.log_image(fig, basename(batch_paths[0]), title)

            inputs, labels = data_processor.process_batch(images, labels)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                loss=f"{running_loss / (i + 1):.4f}",
                acc=f"{100.0 * correct / total:.1f}%",
            )

        epoch_loss = running_loss / iters
        epoch_acc = correct / total
        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_acc)

        tracker.log_metrics(epoch, epoch_loss, epoch_acc)
        logger.info(
            "Epoch %d: loss=%.4f, accuracy=%.1f%%",
            epoch + 1,
            epoch_loss,
            100.0 * epoch_acc,
        )

    tracker.finish()
    logger.info("Training complete.")

    return TrainResult(
        model=model,
        data_processor=data_processor,
        history=history,
        config=config,
    )


def evaluate(
    result: TrainResult,
    *,
    test_dir: Optional[str] = None,
) -> float:
    """Evaluate a trained model on the test set.

    Parameters
    ----------
    result : TrainResult
        The object returned by :func:`train`.
    test_dir : str, optional
        Override the test directory (default: ``config.TESTING_DATA_DIR``).

    Returns
    -------
    float
        Test accuracy (0-1).
    """
    config = result.config
    test_path = test_dir or config.TESTING_DATA_DIR

    test_images = get_image_paths(test_path)
    if not test_images:
        logger.warning("No test images found in %s", test_path)
        return 0.0

    logger.info("Evaluating on %d test images...", len(test_images))

    model = result.model
    data_processor = result.data_processor

    model.eval()
    correct = 0
    total = 0
    used: List[str] = []
    iters = len(test_images) // config.batch_size

    with torch.no_grad():
        for _ in tqdm(range(iters), desc="Evaluating"):
            batch_paths, used = select_random_images(
                test_images, config.batch_size, used
            )
            images, labels = data_processor.get_data_from_paths(batch_paths)
            inputs, labels = data_processor.process_batch(images, labels)

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    logger.info("Test accuracy: %.1f%%", 100.0 * accuracy)
    return accuracy
