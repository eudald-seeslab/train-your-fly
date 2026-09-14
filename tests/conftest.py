import os

import pandas as pd
import pytest


def build_binocular_connectome_data(root) -> str:
    """Write a tiny synthetic ``connectome_data/`` folder with both eyes.

    Neurons 1-6 belong to the left eye, 7-12 to the right eye (four R7 each,
    at the corners of a square so that SciPy Voronoi works, plus one R8 and one
    R1-6 near the middle). Neurons 13-16 are central: two Kenyon cells (one per
    hemisphere), one ``center`` neuron and one with unknown annotations.

    Returns the path of the created ``connectome_data`` folder.
    """
    data_dir = os.path.join(root, "connectome_data")
    os.makedirs(data_dir, exist_ok=True)

    eye_types = ["R7", "R7", "R7", "R7", "R8", "R1-6"]
    left_ids = [str(i) for i in range(1, 7)]
    right_ids = [str(i) for i in range(7, 13)]

    neurons = pd.DataFrame(
        {
            "root_id": left_ids + right_ids + ["13", "14", "15", "16"],
            "cell_type": eye_types * 2 + ["KCapbp-m", "KCapbp-m", "Tm1", None],
            "side": ["left"] * 6 + ["right"] * 6 + ["left", "right", "center", None],
        }
    )
    neurons.to_csv(os.path.join(data_dir, "classification.csv"), index=False)

    # Photoreceptors feed the Kenyon cell of their hemisphere with varying
    # strength; the central neurons form a small chain. Neuron 16 has a single
    # weak connection so that a synapse threshold removes it from the graph.
    connections = pd.DataFrame(
        {
            "pre_root_id": left_ids + right_ids + ["13", "14", "15"],
            "post_root_id": ["13"] * 6 + ["14"] * 6 + ["15", "15", "16"],
            "syn_count": [1, 2, 3, 4, 5, 6] * 2 + [7, 8, 1],
        }
    )
    connections.to_csv(os.path.join(data_dir, "connections.csv"), index=False)

    positions = {
        "x": [0] * 6,
        "y": [0] * 6,
        "z": [0] * 6,
        "PC1": [0] * 6,
        "PC2": [0] * 6,
        "x_axis": [100, 400, 400, 100, 250, 150],
        "y_axis": [100, 100, 400, 400, 250, 250],
        "cell_type": eye_types,
    }
    for eye, ids in (("left", left_ids), ("right", right_ids)):
        pd.DataFrame({"root_id": ids, **positions}).to_csv(
            os.path.join(data_dir, f"{eye}_visual_positions_all_neurons.csv"),
            index=False,
        )

    return data_dir


@pytest.fixture
def binocular_data_dir(tmp_path) -> str:
    return build_binocular_connectome_data(str(tmp_path))
