# FIXME: think about whether we add the randomizers to the main package or keep them in a separate package

import numpy as np
import pandas as pd

from trainyourfly.utils.utils import add_coords
from trainyourfly.utils.randomizers import binned, connection_pruning, pruning, mantain_neuron_wiring_length
from trainyourfly.utils.randomizers.randomizers_helpers import compute_total_synapse_length


def _sample_network():
    """Return small example (connections, neurons) dataframes for tests."""
    # Five neurons arranged in a grid-like fashion for non-zero distances
    neurons = pd.DataFrame(
        {
            "root_id": [1, 2, 3, 4, 5],
            "pos_x": [0, 0, 1, 1, 2],
            "pos_y": [0, 1, 0, 1, 1],
            "pos_z": [0, 0, 0, 0, 0],
        }
    )

    connections = pd.DataFrame(
        {
            "pre_root_id": [1, 1, 1, 2, 2, 3, 3, 4],
            "post_root_id": [2, 3, 4, 3, 4, 4, 5, 5],
            # use modest syn counts so pruning can lower them without hitting 0
            "syn_count": [5, 3, 2, 4, 1, 2, 3, 1],
        }
    )
    return connections, neurons


# -----------------------------------------------------------------------------
# Binned length–preserving shuffler
# -----------------------------------------------------------------------------

def test_binned_length_preservation():
    conns, nc = _sample_network()
    orig_len = compute_total_synapse_length(conns, nc)

    shuffled = binned.create_length_preserving_random_network(
        conns, nc, bins=2, tolerance=0.2
    )

    # Same number of rows & total synapses.
    assert len(shuffled) == len(conns)
    assert shuffled["syn_count"].sum() == conns["syn_count"].sum()

    # Wiring length must be close to the original (≤20 % deviation for small toy dataset)
    new_len = compute_total_synapse_length(shuffled, nc)
    rel_err = abs(new_len - orig_len) / orig_len
    assert rel_err < 0.2


# -----------------------------------------------------------------------------
# Random pruning – binomial synapse removal
# -----------------------------------------------------------------------------

def test_random_pruning_matches_target_length():
    conns, nc = _sample_network()
    orig_len = compute_total_synapse_length(conns, nc)

    target_len = 0.8 * orig_len  # prune to 80 % of the original length

    pruned = pruning.match_wiring_length_with_random_pruning(
        conns,
        nc,
        real_length=target_len,
        tolerance=0.25,  # higher tolerance for tiny integer counts
        random_state=42,
        max_iter=30,
    )

    new_len = compute_total_synapse_length(pruned, nc)
    rel_err = abs(new_len - target_len) / target_len
    assert rel_err <= 0.25
    # syn counts are non-negative integers
    assert (pruned["syn_count"] >= 0).all()
    assert pd.api.types.is_integer_dtype(pruned["syn_count"])


# -----------------------------------------------------------------------------
# Connection pruning – whole-edge removal
# -----------------------------------------------------------------------------

def test_connection_pruning_matches_target_length():
    conns, nc = _sample_network()
    orig_len = compute_total_synapse_length(conns, nc)
    target_len = 0.80 * orig_len  # easier target for tiny graph

    pruned = connection_pruning.match_wiring_length_with_connection_pruning(
        conns,
        nc,
        real_length=target_len,
        tolerance=0.3,  # relax for tiny graph
        random_state=42,
        max_iter=50,
    )

    new_len = compute_total_synapse_length(pruned, nc)
    rel_err = abs(new_len - target_len) / target_len
    assert rel_err <= 0.3
    # number of connections should not exceed original
    assert len(pruned) <= len(conns)


# -----------------------------------------------------------------------------
# Per-neuron wiring-length preservation
# -----------------------------------------------------------------------------

def test_mantain_neuron_wiring_length_preservation():
    conns, nc = _sample_network()
    orig_len = compute_total_synapse_length(conns, nc)

    shuffled = mantain_neuron_wiring_length.mantain_neuron_wiring_length(
        conns,
        nc,
        bins=3,
        min_connections_for_binning=2,
        random_state=1234
    )

    new_len = compute_total_synapse_length(shuffled, nc)
    rel_err = abs(new_len - orig_len) / orig_len
    assert rel_err < 0.2

    # Additionally, check per-neuron outgoing wiring length is roughly preserved
    def per_neuron_wiring(df):
        df_coords = add_coords(df, nc)
        lengths = np.linalg.norm(
            df_coords[["pre_x", "pre_y", "pre_z"]].values
            - df_coords[["post_x", "post_y", "post_z"]].values,
            axis=1,
        )
        return (
            pd.Series(lengths * df["syn_count"].values).groupby(df["pre_root_id"]).sum()
        )

    orig_per = per_neuron_wiring(conns)
    new_per = per_neuron_wiring(shuffled)

    # Align indices and compute relative error per neuron
    new_per = new_per.reindex(orig_per.index, fill_value=0)
    rel_errors = abs(new_per - orig_per) / (orig_per + 1e-9)
    assert (rel_errors < 0.3).all()  # ≤30 % deviation on tiny dataset 