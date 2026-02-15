"""
Simple Training Example for TrainYourFly
=========================================

Usage:
    # With YAML config
    python examples/simple_training.py --config config.yaml

    # With default config pointing at your data
    python examples/simple_training.py --data-dir my_data

    # Quick demo (few epochs)
    python examples/simple_training.py --config config.yaml --epochs 2
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainyourfly import Config, train, evaluate

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train a Drosophila connectome-based neural network"
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--data-dir", "-d", type=str, default="data",
        help="Path to your dataset folder (must contain train/ and test/)",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run evaluation on test set after training",
    )
    args = parser.parse_args()

    # Build config
    if args.config:
        logger.info("Loading config from: %s", args.config)
        config = Config.from_yaml(args.config)
    else:
        config = Config(data_dir=args.data_dir)

    # Apply CLI overrides
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    # Train
    result = train(config)

    # Evaluate
    if args.evaluate:
        evaluate(result)


if __name__ == "__main__":
    main()
