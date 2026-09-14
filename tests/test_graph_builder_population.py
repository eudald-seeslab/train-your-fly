from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import coo_matrix
import torch
from torch import nn

from trainyourfly.config import Config
from trainyourfly.connectome_models.graph_builder import GraphBuilder
from trainyourfly.connectome_models.graph_models import Connectome
from trainyourfly.utils.csv_loader import CSVLoader


def _builder(data_dir, **overrides) -> GraphBuilder:
    config = Config(
        connectome_data_dir=data_dir, device_type="cpu", random_seed=7, **overrides
    )
    return GraphBuilder.from_dataset(
        data_dir=data_dir,
        csv_loader=CSVLoader(use_cache=False),
        rational_cell_types=["KCapbp-m"],
        config=config,
    )


def _cell_types(gb: GraphBuilder) -> list[str]:
    return [gb.cell_type_names[c] for c in gb.node_cell_type]


def _root_ids_in_node_order(gb: GraphBuilder) -> list[str]:
    return gb.root_ids.sort_values("index_id")["root_id"].tolist()


# -----------------------------------------------------------------------------
# min_synapses
# -----------------------------------------------------------------------------

def test_min_synapses_default_keeps_everything(binocular_data_dir):
    gb = _builder(binocular_data_dir)
    assert gb.num_nodes == 16
    assert gb.synaptic_matrix.nnz == 15


def test_min_synapses_drops_weak_connections_and_isolated_neurons(binocular_data_dir):
    gb = _builder(binocular_data_dir, min_synapses=5)

    # Connections with syn_count >= 5: two per eye (R8 and R1-6) plus 13->15, 14->15
    assert gb.synaptic_matrix.nnz == 6
    assert gb.synaptic_matrix.data.min() >= 5

    # Only neurons that still take part in a connection remain: R8 and R1-6 of
    # each eye, both Kenyon cells and neuron 15. Neuron 16 only received a
    # single-synapse connection and is gone, as are all R7.
    kept = sorted(_root_ids_in_node_order(gb), key=int)
    assert kept == ["5", "6", "11", "12", "13", "14", "15"]
    assert gb.num_nodes == 7
    assert gb.node_cell_type.shape == (gb.num_nodes,)
    assert "R7" not in gb.cell_type_names


def test_min_synapses_in_yaml_roundtrip(tmp_path):
    path = tmp_path / "config.yaml"
    Config.create_example(str(path))
    assert "min_synapses" in path.read_text()

    Config(min_synapses=5).to_yaml(str(path))
    assert Config.from_yaml(str(path)).min_synapses == 5


# -----------------------------------------------------------------------------
# Node annotations
# -----------------------------------------------------------------------------

def test_node_annotations_follow_index_id_order(binocular_data_dir):
    gb = _builder(binocular_data_dir)
    root_ids = _root_ids_in_node_order(gb)

    assert gb.cell_type_names == sorted(gb.cell_type_names)
    assert gb.cell_type_names == ["KCapbp-m", "R1-6", "R7", "R8", "Tm1", "Unknown"]
    assert gb.node_cell_type.shape == (16,)
    assert gb.node_side.shape == (16,)

    expected_type = {"13": "KCapbp-m", "14": "KCapbp-m", "15": "Tm1", "16": "Unknown"}
    expected_side = {"13": 0, "14": 1, "15": 2, "16": 2}
    for rid, cell_type, side in zip(root_ids, _cell_types(gb), gb.node_side):
        if int(rid) <= 6:
            assert side == 0
        elif int(rid) <= 12:
            assert side == 1
        else:
            assert cell_type == expected_type[rid]
            assert side == expected_side[rid]


def test_node_annotations_from_neuron_data():
    neurons = pd.DataFrame(
        {
            "root_id": ["a", "b", "c"],
            "cell_type": ["T4a", "T4a", None],
            "side": ["right", "left", "na"],
        }
    )
    connections = pd.DataFrame(
        {"pre_root_id": ["a", "b"], "post_root_id": ["b", "c"], "syn_count": [3, 9]}
    )
    root_ids = pd.DataFrame({"root_id": ["c", "a", "b"], "index_id": [0, 1, 2]})

    gb = GraphBuilder.from_neuron_data(
        neurons, connections, root_ids, device=torch.device("cpu")
    )

    assert gb.cell_type_names == ["T4a", "Unknown"]
    assert _cell_types(gb) == ["Unknown", "T4a", "T4a"]
    assert gb.node_side.tolist() == [2, 1, 0]


def test_node_indices_for_types(binocular_data_dir):
    gb = _builder(binocular_data_dir)
    root_ids = np.array(_root_ids_in_node_order(gb))

    r7 = gb.node_indices_for_types(["R7"])
    assert np.all(np.diff(r7) > 0)
    assert sorted(root_ids[r7].tolist(), key=int) == ["1", "2", "3", "4", "7", "8", "9", "10"]

    r7_left = gb.node_indices_for_types(["R7"], side="left")
    assert sorted(root_ids[r7_left].tolist(), key=int) == ["1", "2", "3", "4"]

    kc_right = gb.node_indices_for_types(["KCapbp-m"], side="right")
    assert root_ids[kc_right].tolist() == ["14"]

    both = gb.node_indices_for_types(["R8", "R1-6"], side="right")
    assert sorted(root_ids[both].tolist(), key=int) == ["11", "12"]

    assert gb.node_indices_for_types(["not-a-type"]).size == 0

    with pytest.raises(ValueError):
        gb.node_indices_for_types(["R7"], side="middle")


# -----------------------------------------------------------------------------
# to_torch_sparse_csr
# -----------------------------------------------------------------------------

def _tiny_builder() -> GraphBuilder:
    # pre -> post with weights: 0->1 (2), 0->2 (5), 1->2 (3), 2->0 (7)
    pre = np.array([0, 0, 1, 2])
    post = np.array([1, 2, 2, 0])
    syn = np.array([2, 5, 3, 7])
    matrix = coo_matrix((syn, (pre, post)), shape=(3, 3))
    return GraphBuilder.from_synaptic_matrix(matrix, device=torch.device("cpu"))


def test_to_torch_sparse_csr_matches_dense_transpose():
    gb = _tiny_builder()
    W = gb.to_torch_sparse_csr(torch.device("cpu"), dtype=torch.float64)

    assert W.layout == torch.sparse_csr
    assert W.dtype == torch.float64
    assert W.shape == (3, 3)

    dense = torch.as_tensor(gb.synaptic_matrix.toarray(), dtype=torch.float64)
    assert torch.equal(W.to_dense(), dense.T)
    assert W.to_dense()[1, 0] == 2  # W[post, pre]

    X = torch.rand(3, 4, dtype=torch.float64)
    assert torch.allclose(W @ X, dense.T @ X)


def test_to_torch_sparse_csr_propagates_like_connectome():
    gb = _tiny_builder()
    config = SimpleNamespace(
        NUM_CONNECTOME_PASSES=1,
        batch_size=1,
        train_edges=False,
        train_neurons=False,
        lambda_func=nn.Identity(),
        neuron_normalization="min_max",
        refined_synaptic_data=False,
        synaptic_limit=False,
        dtype=torch.float32,
        DEVICE=torch.device("cpu"),
        neuron_dropout=0.0,
    )
    model = Connectome(SimpleNamespace(graph_builder=gb), config)

    x = torch.tensor([[1.0], [10.0], [100.0]])
    propagated = model(x, gb.edges.long())

    W = gb.to_torch_sparse_csr(torch.device("cpu"))
    assert torch.allclose(propagated, W @ x)
    # Node 2 receives from nodes 0 (5 synapses) and 1 (3 synapses)
    assert propagated[2].item() == pytest.approx(5 * 1.0 + 3 * 10.0)
