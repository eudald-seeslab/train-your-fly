"""
Training with Weights & Biases tracking
========================================

Same training, but with W&B experiment tracking for metrics, images, and tables.

Prerequisites:
    pip install wandb
    wandb login

Usage:
    python examples/training_with_wandb.py --config config.yaml \
        --wandb-project my-fly-project
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainyourfly import Config, train, evaluate
from trainyourfly.integrations.wandb_tracker import WandBTracker


def main():
    parser = argparse.ArgumentParser(description="Train with W&B tracking")
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--wandb-project", type=str, default="train-your-fly")
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    tracker = WandBTracker(
        project=args.wandb_project,
        group=args.wandb_group,
    )

    result = train(config, tracker=tracker)

    if args.evaluate:
        evaluate(result)


if __name__ == "__main__":
    main()
