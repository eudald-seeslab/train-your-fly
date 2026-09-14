import os

from trainyourfly.utils.train_funcs import (
    construct_synaptic_matrix,
    get_side_decision_making_vector,
)
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import torch
from torch_geometric.data import Data


class GraphBuilder:
    """Utility to convert a sparse synaptic connectivity matrix plus
    per-neuron activations into batched `torch_geometric.data.Data` graphs.

    This is the `connectome` part of the code. It builds a Pytorch Geometric graph from the
    connections between neurons, and initializes it with the per-neuron activations coming from
    neuron_mapper (which acts as the retina).

    Parameters
    ----------
    edges : torch.Tensor
        Edge index tensor of shape ``(2, E)`` with ``int32`` dtype.
    weights : torch.Tensor | None, optional
        Optional edge-weight tensor.  Not yet used in the current pipeline
        but stored for future extensions (e.g. weighted message passing).
    device : torch.device
        Target device for the generated tensors.
    synaptic_matrix : scipy.sparse.coo_matrix | None, optional
        Full synaptic matrix (optional).

    The factories :meth:`from_dataset` and :meth:`from_neuron_data` also set
    per-node annotations aligned with ``root_ids.index_id`` (node ``i`` is
    the neuron with ``index_id == i``):

    ``node_cell_type``
        ``np.ndarray[int]`` of shape ``[num_nodes]``, index into
        ``cell_type_names``.
    ``cell_type_names``
        Sorted unique cell types of the kept neurons (``"Unknown"`` included
        when present).
    ``node_side``
        ``np.ndarray[int]`` of shape ``[num_nodes]``: ``0`` left, ``1`` right,
        ``2`` anything else (center, na, Unknown).
    """

    SIDE_CODES = {"left": 0, "right": 1}
    OTHER_SIDE_CODE = 2

    def __init__(
        self,
        edges: torch.Tensor,
        *,
        device: torch.device,
        weights: torch.Tensor | None = None,
        synaptic_matrix: coo_matrix | None = None,
    ):
        self.edges = edges
        self.device = device
        self.weights = weights
        self.synaptic_matrix = synaptic_matrix  # full sparse matrix (optional)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_synaptic_matrix(cls, syn_matrix: coo_matrix, *, device: torch.device):
        """Create a :class:`GraphBuilder` directly from a SciPy COO matrix."""

        # Stack the row and column indices -> (2, E)
        edges_np = np.vstack((syn_matrix.row, syn_matrix.col)).astype(np.int32)
        edges = torch.as_tensor(edges_np, dtype=torch.int32)

        weights = torch.as_tensor(syn_matrix.data, dtype=torch.float32)

        return cls(edges, device=device, weights=weights, synaptic_matrix=syn_matrix)

    def build_batch(
        self, activation_tensor: torch.Tensor, labels
    ) -> tuple[Data, torch.Tensor]:
        """Create a batched graph for *labels* from *activation_tensor*.

        Parameters
        ----------
        activation_tensor : torch.Tensor
            Shape ``(N_nodes, B)`` where ``B`` is the batch size.
        labels : Sequence[int]
            Integer class labels (length ``B``).
        """
        batch_size = len(labels)
        num_nodes = activation_tensor.shape[0]
        num_edges = self.edges.shape[1]

        x = activation_tensor.t().contiguous().view(-1, 1)

        edge_index_rep = self.edges.to(self.device).repeat(1, batch_size)
        node_offsets = (
            torch.arange(batch_size, device=self.device, dtype=torch.int32) * num_nodes
        ).repeat_interleave(num_edges)
        edge_index_rep = edge_index_rep + node_offsets.unsqueeze(0)

        batch_vec = torch.arange(batch_size, device=self.device).repeat_interleave(
            num_nodes
        )

        inputs = Data(x=x, edge_index=edge_index_rep, batch=batch_vec)
        labels_t = torch.tensor(labels, dtype=torch.long, device=self.device)

        return inputs, labels_t

    @classmethod
    def from_neuron_data(
        cls,
        neuron_classification: "pd.DataFrame",
        connections: "pd.DataFrame",
        root_ids: "pd.DataFrame",
        *,
        log_transform: bool = False,
        device: torch.device,
    ) -> "GraphBuilder":
        """High-level factory that builds the sparse synaptic matrix from
        *neuron_classification* and *connections* and returns both the
        :class:`GraphBuilder` *and* the SciPy COO matrix.

        This moves low-level graph responsibilities out of
        ``DataProcessor``.
        """

        syn_mat = construct_synaptic_matrix(
            neuron_classification, connections, root_ids
        )

        if log_transform:
            syn_mat.data = np.log1p(syn_mat.data)

        builder = cls.from_synaptic_matrix(syn_mat, device=device)

        # Keep useful references
        builder.root_ids = root_ids
        builder.decision_making_vector = get_side_decision_making_vector(
            root_ids,
            neuron_classification["cell_type"].unique().tolist(),
            neuron_classification,
        )
        builder._set_node_annotations(neuron_classification, root_ids)

        return builder

    # ------------------------------------------------------------------
    # Convenience properties & utilities
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        """Return the number of neurons in the graph."""
        if self.synaptic_matrix is None:
            raise AttributeError(
                "GraphBuilder was created without synaptic_matrix info"
            )
        return self.synaptic_matrix.shape[0]

    def to_torch_sparse_csr(
        self, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Return the connectivity as a torch CSR tensor ``W`` of shape
        ``[num_nodes, num_nodes]`` with ``W[post, pre] = syn_count``.

        This is the transpose of ``synaptic_matrix`` (rows = pre, cols = post),
        so that ``W @ X`` for a state ``X`` of shape ``[num_nodes, B]`` sends
        activity from pre- to postsynaptic neurons, exactly like one pass of
        :class:`~trainyourfly.connectome_models.graph_models.Connectome`
        with untrained edges.
        """
        if self.synaptic_matrix is None:
            raise AttributeError(
                "GraphBuilder was created without synaptic_matrix info"
            )

        post_by_pre = self.synaptic_matrix.transpose().tocsr()
        post_by_pre.sort_indices()

        return torch.sparse_csr_tensor(
            torch.as_tensor(post_by_pre.indptr, dtype=torch.int64),
            torch.as_tensor(post_by_pre.indices, dtype=torch.int64),
            torch.as_tensor(post_by_pre.data, dtype=dtype),
            size=post_by_pre.shape,
            device=device,
        )

    def node_indices_for_types(
        self, types: list[str], side: str | None = None
    ) -> np.ndarray:
        """Return the sorted node indices whose cell type is in *types*,
        optionally restricted to one hemisphere (``"left"`` or ``"right"``).
        """
        type_codes = [
            self.cell_type_names.index(t) for t in types if t in self.cell_type_names
        ]
        mask = np.isin(self.node_cell_type, type_codes)

        if side is not None:
            if side not in self.SIDE_CODES:
                raise ValueError(
                    f"side must be one of {sorted(self.SIDE_CODES)}, got {side!r}"
                )
            mask &= self.node_side == self.SIDE_CODES[side]

        return np.flatnonzero(mask)

    def _set_node_annotations(
        self, neuron_classification: pd.DataFrame, root_ids: pd.DataFrame
    ) -> None:
        """Populate ``node_cell_type``, ``cell_type_names`` and ``node_side``
        in ``index_id`` order from the classification table."""

        annotated = (
            root_ids[["root_id", "index_id"]]
            .merge(
                neuron_classification[["root_id", "cell_type", "side"]],
                on="root_id",
                how="left",
            )
            .fillna({"cell_type": "Unknown", "side": "Unknown"})
            .sort_values("index_id")
        )
        index_id = annotated["index_id"].to_numpy()
        num_nodes = len(root_ids)

        self.cell_type_names = sorted(annotated["cell_type"].unique().tolist())
        type_codes = pd.Categorical(
            annotated["cell_type"], categories=self.cell_type_names
        ).codes

        side_codes = (
            annotated["side"]
            .map(self.SIDE_CODES)
            .fillna(self.OTHER_SIDE_CODE)
            .astype(int)
            .to_numpy()
        )

        self.node_cell_type = np.empty(num_nodes, dtype=np.int64)
        self.node_cell_type[index_id] = type_codes
        self.node_side = np.empty(num_nodes, dtype=np.int64)
        self.node_side[index_id] = side_codes

    @staticmethod
    def shuffle_synaptic_matrix(synaptic_matrix: coo_matrix) -> coo_matrix:
        """Randomly permute the *postsynaptic* column indices to produce a
        shuffled version of the connectivity matrix (used for randomisation
        baselines)."""

        shuffled_col = np.random.permutation(synaptic_matrix.col)
        shuffled = coo_matrix(
            (synaptic_matrix.data, (synaptic_matrix.row, shuffled_col)),
            shape=synaptic_matrix.shape,
        )
        shuffled.sum_duplicates()
        return shuffled

    @classmethod
    def from_dataset(
        cls,
        *,
        data_dir: str,
        csv_loader,
        rational_cell_types: list[str],
        config,
    ):
        """High-level factory that reads CSVs, builds matrix and returns a GraphBuilder."""

        neuron_classification = cls._load_neurons(
            data_dir,
            csv_loader,
            config.filtered_celltypes,
            config.filtered_fraction,
        )

        # Config objects may be plain namespaces that lack optional attributes.
        min_synapses = getattr(config, "min_synapses", None)

        connections = cls._load_connections(
            data_dir,
            csv_loader,
            config.refined_synaptic_data,
            config.randomization_strategy,
            min_synapses=min_synapses,
        )

        root_ids = cls._compute_root_ids(neuron_classification, connections)

        syn_mat = construct_synaptic_matrix(
            neuron_classification, connections, root_ids
        )

        if config.log_transform_weights:
            syn_mat.data = np.log1p(syn_mat.data)

        builder = cls.from_synaptic_matrix(syn_mat, device=config.DEVICE)
        builder.root_ids = root_ids
        builder.decision_making_vector = get_side_decision_making_vector(
            root_ids,
            rational_cell_types,
            neuron_classification,
        )
        builder._set_node_annotations(neuron_classification, root_ids)

        return builder

    # ------------------------------------------------------------------
    # Internal CSV helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_neurons(data_dir, csv_loader, filtered_cell_types, filtered_fraction):
        df = csv_loader.read_csv(
            os.path.join(data_dir, "classification.csv"),
            usecols=["root_id", "cell_type", "side"],
            dtype={"root_id": "string"},
        ).fillna("Unknown")

        if filtered_cell_types:
            df = df[~df["cell_type"].isin(filtered_cell_types)]

        if filtered_fraction is not None:
            protected = df[df["cell_type"].isin(["R1-6", "R7", "R8"])]
            non_protected = df[~df["cell_type"].isin(["R1-6", "R7", "R8"])]
            non_protected = non_protected.sample(
                frac=filtered_fraction, random_state=1714
            )
            df = pd.concat([protected, non_protected])
        return df

    @staticmethod
    def _load_connections(
        data_dir, csv_loader, refined, strategy, min_synapses: int | None = None
    ):
        tag = "_refined" if refined else ""
        if strategy is not None:
            tag += f"_random_{strategy}"
        fname = os.path.join(data_dir, f"connections{tag}.csv")
        conns = csv_loader.read_csv(
            fname,
            dtype={
                "pre_root_id": "string",
                "post_root_id": "string",
                "syn_count": np.int32,
            },
        )
        grouped = (
            conns.groupby(["pre_root_id", "post_root_id"], as_index=False)["syn_count"]
            .sum()
        )
        if min_synapses is not None:
            grouped = grouped[grouped["syn_count"] >= min_synapses]
        return grouped.sort_values(["pre_root_id", "post_root_id"])

    @staticmethod
    def _compute_root_ids(classification, connections):
        neurons = classification[
            classification["root_id"].isin(connections["pre_root_id"])
            | classification["root_id"].isin(connections["post_root_id"])
        ]
        return (
            neurons.reset_index(drop=True)
            .reset_index()[["root_id", "index"]]
            .rename(columns={"index": "index_id"})
        )
