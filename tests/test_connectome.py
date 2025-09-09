import torch
from torch import nn
from torch_geometric.data import Data

from connectome.core.graph_models import Connectome


class DummyDataProcessor:
    def __init__(self):
        # Two-node toy graph with two edges
        import numpy as np
        from scipy.sparse import coo_matrix

        rows = np.array([0, 1])
        cols = np.array([1, 0])
        data = np.array([1, 1])
        self.synaptic_matrix = coo_matrix((data, (rows, cols)), shape=(2, 2))
        # Minimal GraphBuilder stub
        from connectome.core.graph_builder import GraphBuilder

        edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
        self.graph_builder = GraphBuilder(edges, device=torch.device("cpu"), synaptic_matrix=self.synaptic_matrix)
        self.number_of_synapses = self.graph_builder.num_nodes


class DummyConfig:
    NUM_CONNECTOME_PASSES = 1
    batch_size = 1
    train_edges = False
    train_neurons = False
    lambda_func = nn.Identity()
    neuron_normalization = "min_max"
    refined_synaptic_data = False
    synaptic_limit = False
    dtype = torch.float32
    DEVICE = torch.device("cpu")
    neuron_dropout = 0.0


def test_connectome_forward():
    dp = DummyDataProcessor()
    config = DummyConfig()
    model = Connectome(dp, config)

    # Define graph
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    x = torch.rand(2, 1, dtype=torch.float32)  # two nodes

    out = model(x, edge_index)
    # Output should keep same shape as input
    assert out.shape == x.shape 