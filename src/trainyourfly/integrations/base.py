"""
Base interface for experiment tracking.

TrainYourFly is agnostic to the experiment tracking tool you use.
This module defines a lightweight Protocol that any tracker must satisfy,
plus a NullTracker that silently discards everything (the default).

To integrate your preferred tool, implement the ExperimentTracker protocol
and pass the instance to your training loop.  Ready-made implementations
live in this package:

    from trainyourfly.integrations.wandb_tracker import WandBTracker

See ``examples/training_with_wandb.py`` for a complete example.
"""

from __future__ import annotations

from typing import Any, List, Optional, Protocol, runtime_checkable

import pandas as pd
from matplotlib.figure import Figure


@runtime_checkable
class ExperimentTracker(Protocol):
    """Protocol that any experiment tracker must satisfy.

    All methods are optional in the sense that a NullTracker (which does
    nothing) is provided as the default.  Implement only the methods you
    care about in your own tracker.
    """

    def initialize(self, config: Any) -> None:
        """Called once before training starts."""
        ...

    def log_metrics(
        self,
        epoch: int,
        loss: float,
        accuracy: float,
        *,
        task: Optional[str] = None,
    ) -> None:
        """Log scalar metrics for a training step / epoch."""
        ...

    def log_image(
        self,
        figure: Figure,
        name: str,
        title: str,
        *,
        task: Optional[str] = None,
    ) -> None:
        """Log a matplotlib Figure (e.g. Voronoi overlay, neuron activations)."""
        ...

    def log_dataframe(self, df: pd.DataFrame, title: str) -> None:
        """Log a pandas DataFrame (e.g. per-image predictions)."""
        ...

    def log_validation(
        self,
        loss: float,
        accuracy: float,
        results_df: pd.DataFrame,
        plots: List[Figure],
        *,
        task: Optional[str] = None,
    ) -> None:
        """Log validation / test results at the end of an epoch."""
        ...

    def finish(self) -> None:
        """Called once after training completes (success or failure)."""
        ...


class NullTracker:
    """Default tracker that silently discards all events.

    Use this when you don't need experiment tracking::

        from trainyourfly.integrations import NullTracker
        tracker = NullTracker()
    """

    def initialize(self, config: Any) -> None:
        pass

    def log_metrics(
        self,
        epoch: int,
        loss: float,
        accuracy: float,
        *,
        task: Optional[str] = None,
    ) -> None:
        pass

    def log_image(
        self,
        figure: Figure,
        name: str,
        title: str,
        *,
        task: Optional[str] = None,
    ) -> None:
        pass

    def log_dataframe(self, df: pd.DataFrame, title: str) -> None:
        pass

    def log_validation(
        self,
        loss: float,
        accuracy: float,
        results_df: pd.DataFrame,
        plots: List[Figure],
        *,
        task: Optional[str] = None,
    ) -> None:
        pass

    def finish(self) -> None:
        pass
