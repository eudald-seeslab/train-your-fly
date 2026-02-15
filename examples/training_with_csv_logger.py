"""
Training with a custom CSV tracker
====================================

Demonstrates how to write your own tracker. This one writes metrics to a
CSV file and saves plots to disk -- no external services required.

This pattern works for any tool: MLflow, TensorBoard, Neptune, etc.

Usage:
    python examples/training_with_csv_logger.py --config config.yaml
"""

import csv
import logging
import os
import sys
from typing import Any, List, Optional

import pandas as pd
from matplotlib.figure import Figure

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainyourfly import Config, train, evaluate

logger = logging.getLogger(__name__)


class CSVTracker:
    """Minimal tracker that writes metrics to CSV and saves plots to disk."""

    def __init__(self, output_dir: str = "logs"):
        self.output_dir = output_dir
        self._csv_file = None
        self._writer = None

    def initialize(self, config: Any) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        self._csv_file = open(
            os.path.join(self.output_dir, "metrics.csv"), "w", newline=""
        )
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(["epoch", "loss", "accuracy", "task"])

    def log_metrics(self, epoch: int, loss: float, accuracy: float, *, task: Optional[str] = None) -> None:
        self._writer.writerow([epoch, f"{loss:.6f}", f"{accuracy:.6f}", task or ""])
        self._csv_file.flush()

    def log_image(self, figure: Figure, name: str, title: str, *, task: Optional[str] = None) -> None:
        img_dir = os.path.join(self.output_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        figure.savefig(os.path.join(img_dir, f"{title}_{name}.png"), dpi=150, bbox_inches="tight")

    def log_dataframe(self, df: pd.DataFrame, title: str) -> None:
        df.to_csv(os.path.join(self.output_dir, f"{title}.csv"), index=False)

    def log_validation(self, loss: float, accuracy: float, results_df: pd.DataFrame, plots: List[Figure], *, task: Optional[str] = None) -> None:
        self.log_metrics(-1, loss, accuracy, task=task)
        self.log_dataframe(results_df, "validation_results")
        for i, fig in enumerate(plots):
            self.log_image(fig, str(i), "validation_plot", task=task)

    def finish(self) -> None:
        if self._csv_file:
            self._csv_file.close()
        logger.info("Logs saved to %s/", self.output_dir)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train with CSV tracking")
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    tracker = CSVTracker(output_dir=args.log_dir)

    result = train(config, tracker=tracker)

    if args.evaluate:
        evaluate(result)


if __name__ == "__main__":
    main()
