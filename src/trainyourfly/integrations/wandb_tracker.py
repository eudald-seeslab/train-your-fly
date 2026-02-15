"""
Weights & Biases integration for TrainYourFly.

This module provides a WandBTracker that implements the ExperimentTracker
protocol using Weights & Biases (https://wandb.ai).

Usage::

    from trainyourfly.integrations.wandb_tracker import WandBTracker

    tracker = WandBTracker(project="my-fly-project")
    tracker.initialize(config)
    # ... training loop ...
    tracker.finish()

Requires the ``wandb`` package::

    pip install wandb
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

try:
    import wandb
    from wandb import AlertLevel
except ImportError:
    raise ImportError(
        "The 'wandb' package is required for WandBTracker.\n"
        "Install it with:  pip install wandb"
    )

import pandas as pd
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class WandBTracker:
    """Weights & Biases tracker implementing the ExperimentTracker protocol.

    Parameters
    ----------
    project : str
        W&B project name.
    group : str, optional
        W&B run group (e.g. the dataset name). If ``None`` the run is
        ungrouped.
    enabled : bool
        Set to ``False`` to keep the tracker instantiated but silent
        (useful for quick debugging without commenting out tracker calls).
    """

    def __init__(
        self,
        project: str,
        group: Optional[str] = None,
        enabled: bool = True,
    ):
        self.project = project
        self.group = group
        self.enabled = enabled
        self._initialized = False

    # ------------------------------------------------------------------
    # ExperimentTracker protocol
    # ------------------------------------------------------------------

    def initialize(self, config: Any) -> None:
        """Start a W&B run with the given configuration object."""
        if not self.enabled or self._initialized:
            return
        config_dict = self._config_to_dict(config)
        wandb.init(
            project=self.project,
            config=config_dict,
            group=self.group,
        )
        self._initialized = True

    def log_metrics(
        self,
        epoch: int,
        loss: float,
        accuracy: float,
        *,
        task: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        suffix = f" {task}" if task else ""
        try:
            wandb.log(
                {
                    "epoch": epoch,
                    f"loss{suffix}": loss,
                    f"accuracy{suffix}": accuracy,
                }
            )
        except Exception as e:
            logger.warning("Error logging metrics to W&B: %s. Continuing...", e)

    def log_image(
        self,
        figure: Figure,
        name: str,
        title: str,
        *,
        task: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        suffix = f" {task}" if task else ""
        wandb.log(
            {f"{title} image{suffix}": wandb.Image(figure, caption=f"{title} {name}")}
        )

    def log_dataframe(self, df: pd.DataFrame, title: str) -> None:
        if not self.enabled:
            return
        wandb.log({title: wandb.Table(dataframe=df)})

    def log_validation(
        self,
        loss: float,
        accuracy: float,
        results_df: pd.DataFrame,
        plots: List[Figure],
        *,
        task: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        suffix = f" {task}" if task else ""
        plot_dict = {
            f"Plot {i}{suffix}": wandb.Image(plot) for i, plot in enumerate(plots)
        }
        wandb.log(
            {
                f"Validation loss{suffix}": loss,
                f"Validation accuracy{suffix}": accuracy,
                f"Validation results{suffix}": wandb.Table(dataframe=results_df),
            }
            | plot_dict
        )

    def finish(self) -> None:
        if self.enabled:
            wandb.finish()
            self._initialized = False

    # ------------------------------------------------------------------
    # W&B-specific helpers (not part of the protocol)
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        """Return the current W&B run id, or a placeholder."""
        try:
            return wandb.run.id
        except AttributeError:
            return "NO_RUN_ID"

    def send_alert(self, message: str) -> None:
        """Send a W&B alert (e.g. on crash)."""
        if self.enabled:
            wandb.alert(
                title=f"Error in run at {self.project}",
                text=message,
                level=AlertLevel.ERROR,
            )

    def initialize_sweep(self, sweep_config: dict) -> str:
        """Create a W&B sweep and return the sweep id."""
        return wandb.sweep(sweep_config, project=self.project)

    def start_agent(self, sweep_id: str, func) -> None:
        """Start a W&B sweep agent."""
        wandb.agent(sweep_id, function=func)

    @property
    def sweep_config(self):
        """Access to the live W&B sweep config."""
        return wandb.config

    def update_full_config(self, config: Any) -> None:
        """Push all serialisable attributes from *config* to the live W&B
        run so they appear in the run's configuration panel."""
        if not self.enabled:
            return
        full_cfg = self._config_to_dict(config)
        try:
            wandb.config.update(full_cfg, allow_val_change=True)
        except Exception as e:
            logger.warning("Error updating config in W&B: %s. Continuing...", e)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _config_to_dict(config: Any) -> dict:
        """Convert a config object to a serialisable dict."""
        data = {}
        source = config.__dict__ if hasattr(config, "__dict__") else {}
        for k, v in source.items():
            if k.startswith("_") or callable(v):
                continue
            if v is None or isinstance(v, (int, float, bool, str)):
                data[k] = v
            elif isinstance(v, (list, tuple)):
                if all(isinstance(item, (int, float, bool, str)) for item in v):
                    data[k] = list(v)
        return data
