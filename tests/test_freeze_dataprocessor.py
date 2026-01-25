import hashlib
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _build_minimal_adult_data(root) -> None:
    """Create a *very* small `connectome_data/` folder with the CSV files that
    ``DataProcessor`` needs to start up.  The data are synthetic but
    deterministic and tiny so that the test runs in <1 s on CPU.

    Note: the connectome codebase now expects CSVs to live under the
    directory defined by ``configs.config.CONNECTOME_DATA_DIR`` which is
    ``"connectome_data"``.  The freeze test therefore builds that folder
    hierarchy so the refactored ``DataProcessor`` can load the files
    without having to maintain backward-compatibility shims.
    """

    data_dir = os.path.join(root, "connectome_data")
    os.makedirs(data_dir, exist_ok=True)

    # 6 neurons, enough R7 cells (≥4) so that SciPy Voronoi does not complain.
    neurons = pd.DataFrame(
        {
            "root_id": ["1", "2", "3", "4", "5", "6"],
            "cell_type": ["R7", "R7", "R7", "R7", "R8", "R1-6"],
            "side": ["right"] * 6,
        }
    )
    neurons.to_csv(os.path.join(data_dir, "classification.csv"), index=False)

    # Simple ring connections + one R8↔R1-6 pair.
    connections = pd.DataFrame(
        {
            "pre_root_id": ["1", "2", "3", "4", "5", "6"],
            "post_root_id": ["2", "3", "4", "1", "6", "5"],
            "syn_count": [1, 1, 1, 1, 1, 1],
        }
    )

    connections.to_csv(os.path.join(data_dir, "connections.csv"), index=True)

    # Minimal spatial layout (square for R7 centres + two extra neurons).
    visual = pd.DataFrame(
        {
            "root_id": ["1", "2", "3", "4", "5", "6"],
            "x": [0] * 6,
            "y": [0] * 6,
            "z": [0] * 6,
            "PC1": [0] * 6,
            "PC2": [0] * 6,
            "x_axis": [100, 400, 400, 100, 250, 150],
            "y_axis": [100, 100, 400, 400, 250, 250],
            "cell_type": ["R7", "R7", "R7", "R7", "R8", "R1-6"],
        }
    )
    visual.to_csv(os.path.join(data_dir, "right_visual_positions_all_neurons.csv"), index=False)


# -----------------------------------------------------------------------------
# Freeze-test
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("device", ["cpu"])  # keep the test lightweight
def test_dataprocessor_freeze(monkeypatch, tmp_path, device: str):
    """Golden-master test for the *current* behaviour of DataProcessor.process_batch.

    On first run the test creates a fixture file containing the SHA-256 hash of
    the batch.  Subsequent runs must reproduce the exact same hash, otherwise
    the refactor changed observable behaviour.
    """

    # ------------------------------------------------------------------
    # Create synthetic adult_data/ and point PROJECT_ROOT there
    # ------------------------------------------------------------------
    from importlib import import_module

    proj_root = os.path.join(tmp_path, "project_root")
    _build_minimal_adult_data(proj_root)

    # Create empty images directory expected by configs.config
    img_train_dir = os.path.join(proj_root, "images", "one_to_ten", "train", "class0")
    os.makedirs(img_train_dir, exist_ok=True)

    # Write a 1×1 dummy PNG so PIL can open it
    Image.new("RGB", (1, 1), color="white").save(os.path.join(img_train_dir, "dummy.png"))

    paths_mod = import_module("paths")
    monkeypatch.setattr(paths_mod, "PROJECT_ROOT", proj_root, raising=True)

    # Delay heavy imports until after we patched PROJECT_ROOT
    from connectome.core.data_processing import DataProcessor

    # ------------------------------------------------------------------
    # Deterministic config stub
    # ------------------------------------------------------------------
    cfg = SimpleNamespace(
        random_seed=0,
        new_connectome=False,
        rational_cell_types=[],
        filtered_celltypes=[],
        filtered_fraction=None,
        refined_synaptic_data=False,
        randomization_strategy=None,
        log_transform_weights=False,
        voronoi_criteria="R7",
        eye="right",
        neurons="all",
        inhibitory_r7_r8=False,
        dtype=torch.float32,
        DEVICE=torch.device(device),
        CLASSES=["class0", "class1"],
    )

    dp = DataProcessor(cfg)

    # ------------------------------------------------------------------
    # Build a *tiny* deterministic image batch (2×64×64 RGB)
    # ------------------------------------------------------------------
    np.random.seed(0)
    imgs = (np.random.randint(0, 256, size=(2, 64, 64, 3)).astype(np.uint8))
    labels = [0, 1]

    batch, y = dp.process_batch(imgs, labels)

    # ------------------------------------------------------------------
    # Compute SHA-256 digest of critical tensors
    # ------------------------------------------------------------------
    def _tensor_digest(t: torch.Tensor):
        return t.detach().cpu().numpy().tobytes()

    m = hashlib.sha256()
    for tensor in (batch.x, batch.edge_index, batch.batch, y):
        m.update(_tensor_digest(tensor))
    digest = m.hexdigest()

    # ------------------------------------------------------------------
    # Compare with frozen value (create if missing)
    # ------------------------------------------------------------------
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)
    ref_file = os.path.join(fixtures_dir, "graph_batch_v1.sha256")

    if not os.path.exists(ref_file):
        with open(ref_file, "w") as f:
            f.write(digest)
        pytest.skip("Frozen batch hash generated – re-run tests to enable freeze check.")

    with open(ref_file, "r") as f:
        ref_digest = f.read().strip()
    assert digest == ref_digest, (
        "DataProcessor output changed!  If this is intentional, delete "
        f"{ref_file} and re-run the tests to freeze the new behaviour."
    )
