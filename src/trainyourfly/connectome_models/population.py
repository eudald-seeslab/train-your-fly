"""The message passing of :class:`Connectome`, for many brains at once."""

from typing import Callable, Optional

import torch
from torch import nn

from trainyourfly.connectome_models.graph_models import Connectome
from trainyourfly.connectome_models.graph_models_helpers import log_norm, mean_norm, min_max_norm


class PopulationConnectome(nn.Module):
    """Run the connectome for a population of independent brains, inference only.

    :class:`Connectome` batches by repeating the edge list once per sample
    (:meth:`GraphBuilder.build_batch`), which suits the handful of images of a
    training batch and cannot hold hundreds of brains: fifteen million edges
    times the population. Here the state is ``[num_nodes, B]``, one column per
    brain, and a pass is one sparse matmul with ``W[post, pre]``.

    The arithmetic is that of :class:`Connectome`, pass by pass: the same edge
    weights (synapse counts times the squashed gains, when the edges were
    trained), the same normalisation and thresholds (when the neurons were).
    Neurons keep no state between calls. ``tests/test_population_connectome.py``
    checks the two against each other in every training regime.

    Parameters
    ----------
    weight : torch.Tensor
        Sparse CSR ``[num_nodes, num_nodes]`` with ``weight[post, pre]`` the
        effective weight of the connection, as
        :meth:`GraphBuilder.to_torch_sparse_csr` returns it for untrained edges.
    num_passes : int
        Number of message-passing steps (``NUM_CONNECTOME_PASSES``).
    lambda_func : callable, optional
        Activation function applied at every pass, after the normalisation
        (``Config.lambda_func``). ``None`` is the regime in which a pass is the
        weighted sum alone (``train_neurons = activate_neurons = False``).
    neuron_normalization, normalization_scale
        Normalisation of the input before the activation, as in ``Config``:
        ``"min_max"``, ``"log1p"`` or ``"mean"``.
    threshold : torch.Tensor, optional
        Per-neuron activation thresholds ``[num_nodes]`` of a model trained
        with ``train_neurons``; ``None`` is thresholds of zero.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        num_passes: int,
        *,
        lambda_func: Optional[Callable] = None,
        neuron_normalization: str = "min_max",
        normalization_scale: float = 3.0,
        threshold: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if weight.layout != torch.sparse_csr:
            raise ValueError("weight must be a sparse CSR tensor, W[post, pre]")
        if threshold is not None and lambda_func is None:
            raise ValueError("thresholds need the activation function they were trained with")
        # 32-bit indices: the matmul reads fewer bytes per edge.
        self.register_buffer(
            "weight",
            torch.sparse_csr_tensor(
                weight.crow_indices().int(),
                weight.col_indices().int(),
                weight.values(),
                weight.shape,
            ),
            persistent=False,
        )
        self.register_buffer("threshold", threshold, persistent=False)
        self.num_passes = num_passes
        self.lambda_func = lambda_func
        self.neuron_normalization = neuron_normalization
        self.normalization_scale = normalization_scale

    @property
    def num_nodes(self) -> int:
        return self.weight.shape[0]

    @classmethod
    def from_graph_builder(cls, graph_builder, config, device=None) -> "PopulationConnectome":
        """The untrained connectome: the synapse counts as they are (the
        classifier-only regime, ``train_edges = train_neurons = False``), with
        the normalisation and activation of ``config`` if it asks for them
        (``activate_neurons``)."""
        device = config.DEVICE if device is None else device
        activate = getattr(config, "activate_neurons", False)
        return cls(
            graph_builder.to_torch_sparse_csr(device, config.dtype),
            config.NUM_CONNECTOME_PASSES,
            lambda_func=config.lambda_func if activate else None,
            neuron_normalization=config.neuron_normalization,
            normalization_scale=getattr(config, "normalization_scale", 3.0),
        )

    @classmethod
    def from_connectome(cls, connectome: Connectome, graph_builder) -> "PopulationConnectome":
        """A trained (or freshly initialised) :class:`Connectome`: its edge
        gains folded into the weights, its thresholds kept."""
        edge_weight = connectome.edge_weight.detach()
        if connectome.train_edges:
            edge_weight = edge_weight * connectome.edge_activation_func(
                connectome.edge_weight_multiplier.detach()
            )
        # ``edge_weight`` follows the COO order of ``synaptic_matrix`` (rows pre, columns post).
        matrix = graph_builder.synaptic_matrix
        post_by_pre = torch.sparse_coo_tensor(
            torch.as_tensor(
                [matrix.col.tolist(), matrix.row.tolist()], device=edge_weight.device
            ),
            edge_weight,
            matrix.shape[::-1],
        ).coalesce().to_sparse_csr()
        threshold = None
        if connectome.train_neurons:
            threshold = connectome.neuron_activation_threshold.detach().abs()
        return cls(
            post_by_pre,
            connectome.num_passes,
            lambda_func=connectome.lambda_func if connectome.activate_neurons else None,
            neuron_normalization=connectome.neuron_normalization,
            normalization_scale=connectome.normalization_scale,
            threshold=threshold,
        )

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        pre_gain: Optional[torch.Tensor] = None,
        post_gain: Optional[torch.Tensor] = None,
        on_pass: Optional[Callable[[int, torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        """Propagate ``x`` (``[num_nodes, B]``, the receptor activations of ``B``
        brains) through ``num_passes`` passes and return the final state.

        ``pre_gain`` and ``post_gain`` (``[num_nodes, B]``, optional) make the
        brains differ: every pass, a neuron's output is multiplied by its
        ``pre_gain`` and its summed input by its ``post_gain``. ``on_pass(k,
        state)`` is called with the state after each pass ``k`` (from 1), for
        whoever wants to watch the activity spread.
        """
        for k in range(1, self.num_passes + 1):
            if pre_gain is not None:
                x = x * pre_gain
            x = torch.mm(self.weight, x)
            if post_gain is not None:
                x = x * post_gain
            if self.lambda_func is not None:
                # `Connectome.update`, whose samples are rows; here they are columns.
                x = x.t()
                if self.neuron_normalization == "min_max":
                    x = min_max_norm(x)
                elif self.neuron_normalization == "log1p":
                    x = log_norm(x)
                elif self.neuron_normalization == "mean":
                    x = mean_norm(x, self.normalization_scale)
                if self.threshold is not None:
                    x = x - self.threshold
                x = self.lambda_func(x).t()
            if on_pass is not None:
                on_pass(k, x)
        return x
