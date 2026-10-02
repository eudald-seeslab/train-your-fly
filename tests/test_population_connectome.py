from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import coo_matrix
import torch
from torch.nn.functional import leaky_relu

from trainyourfly.connectome_models.graph_builder import GraphBuilder
from trainyourfly.connectome_models.graph_models import Connectome
from trainyourfly.connectome_models.population import PopulationConnectome

NUM_NODES, NUM_EDGES, BATCH = 30, 200, 4


def _builder(signed: bool = False) -> GraphBuilder:
    rng = np.random.default_rng(1714)
    pairs = rng.choice(NUM_NODES * NUM_NODES, size=NUM_EDGES, replace=False)
    pre, post = np.divmod(pairs, NUM_NODES)
    syn = rng.integers(1, 20, size=NUM_EDGES).astype(np.float32)
    if signed:
        syn *= rng.choice([-1.0, 1.0], size=NUM_EDGES).astype(np.float32)
    matrix = coo_matrix((syn, (pre, post)), shape=(NUM_NODES, NUM_NODES))
    return GraphBuilder.from_synaptic_matrix(matrix, device=torch.device("cpu"))


def _config(**overrides) -> SimpleNamespace:
    config = dict(
        NUM_CONNECTOME_PASSES=3,
        batch_size=BATCH,
        train_edges=False,
        train_neurons=False,
        lambda_func=leaky_relu,
        neuron_normalization="min_max",
        refined_synaptic_data=False,
        synaptic_limit=True,
        dtype=torch.float32,
        DEVICE=torch.device("cpu"),
        neuron_dropout=0.0,
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def _through_connectome(model: Connectome, gb: GraphBuilder, x: torch.Tensor) -> torch.Tensor:
    """Run ``x`` (``[num_nodes, B]``) through the PyG model, as training does."""
    data, _ = gb.build_batch(x, [0] * x.shape[1])
    out = model.eval()(data.x, data.edge_index.long())
    return out.view(x.shape[1], -1).t()


@pytest.mark.parametrize(
    "regime",
    [
        dict(),
        dict(train_edges=True),
        dict(train_edges=True, refined_synaptic_data=True),
        dict(train_edges=True, synaptic_limit=False),
        dict(train_neurons=True),
        dict(train_neurons=True, neuron_normalization="log1p"),
        dict(train_edges=True, train_neurons=True),
    ],
)
def test_population_forward_is_the_connectome_forward(regime):
    torch.manual_seed(7)
    gb = _builder(signed=regime.get("refined_synaptic_data", False))
    config = _config(**regime)
    model = Connectome(SimpleNamespace(graph_builder=gb), config)
    population = PopulationConnectome.from_connectome(model, gb)

    x = torch.rand(NUM_NODES, BATCH)
    expected = _through_connectome(model, gb, x)
    assert torch.allclose(population(x), expected, rtol=1e-4, atol=1e-5)
    assert population.num_passes == 3 and population.num_nodes == NUM_NODES


def test_untrained_population_is_the_synapse_counts():
    gb = _builder()
    config = _config()
    population = PopulationConnectome.from_graph_builder(gb, config)
    model = Connectome(SimpleNamespace(graph_builder=gb), config)

    x = torch.rand(NUM_NODES, BATCH)
    assert torch.allclose(population(x), _through_connectome(model, gb, x), rtol=1e-4)
    # One brain does not depend on who else is in the population.
    assert torch.allclose(population(x[:, :1]), population(x)[:, :1], rtol=1e-5)


def test_gains_scale_outputs_and_inputs_every_pass():
    gb = _builder()
    population = PopulationConnectome.from_graph_builder(gb, _config(NUM_CONNECTOME_PASSES=2))
    W = gb.to_torch_sparse_csr(torch.device("cpu")).to_dense()

    x = torch.rand(NUM_NODES, BATCH)
    pre = torch.rand(NUM_NODES, BATCH) + 0.5
    post = torch.rand(NUM_NODES, BATCH) + 0.5
    one_pass = lambda state: post * (W @ (pre * state))
    assert torch.allclose(population(x, pre_gain=pre, post_gain=post), one_pass(one_pass(x)), rtol=1e-4)
    # Unit gains are no gains.
    ones = torch.ones(NUM_NODES, BATCH)
    assert torch.allclose(population(x, pre_gain=ones, post_gain=ones), population(x))


def test_on_pass_sees_every_step():
    gb = _builder()
    population = PopulationConnectome.from_graph_builder(gb, _config())
    seen = []
    out = population(torch.rand(NUM_NODES, 2), on_pass=lambda k, state: seen.append((k, state.clone())))
    assert [k for k, _ in seen] == [1, 2, 3]
    assert torch.equal(seen[-1][1], out)


def test_population_needs_csr_and_an_activation_for_thresholds():
    gb = _builder()
    W = gb.to_torch_sparse_csr(torch.device("cpu"))
    with pytest.raises(ValueError, match="CSR"):
        PopulationConnectome(W.to_dense(), 3)
    with pytest.raises(ValueError, match="activation"):
        PopulationConnectome(W, 3, threshold=torch.zeros(NUM_NODES))
