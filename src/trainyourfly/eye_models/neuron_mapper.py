from __future__ import annotations

import numpy as np
import pandas as pd
import torch


class NeuronMapper:
    """Map per-cell colour averages to per-neuron activations.

    Acts as the retina stage: converts ommatidia colour information into
    neuron-specific activation values, respecting cell-type–specific channel
    mappings and optional R7/R8 inhibition.
    """

    # Channel mapping identical to the original implementation
    _CHANNEL_MAP = {"R1-6": 3, "R7": 2, "R8p": 1, "R8y": 0}

    def __init__(
        self,
        root_ids: pd.DataFrame,
        tesselated_neurons: pd.DataFrame,
        *,
        device: torch.device,
        dtype: torch.dtype,
        inhibitory_r7_r8: bool = False,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.inhibitory_r7_r8 = inhibitory_r7_r8

        self._build_mappings(root_ids, tesselated_neurons)

    def activations_from_voronoi_means(self, voronoi_means: torch.Tensor) -> torch.Tensor:  # (B, C_cells, 4) -> (N_neurons, B)
        """Map per-cell colour averages to per-neuron activations.

        Parameters
        ----------
        voronoi_means : torch.Tensor
            Shape ``(B, num_cells, 4)`` with channel order ``r, g, b, mean``, where B is the batch size.
        """

        vm = voronoi_means.to(self.device)

        r, g, b, m = vm[..., 0], vm[..., 1], vm[..., 2], vm[..., 3]

        if self.inhibitory_r7_r8:
            mask = b > torch.maximum(r, g)
            r = torch.where(mask, 0, r)
            g = torch.where(mask, 0, g)
            b = torch.where(mask, b, 0)

        channels = torch.stack([r, g, b, m], dim=-1)  # B, cells, 4

        valid_neuron_idx = self._valid_neuron_idx
        cell_indices = self._cell_idx_t[valid_neuron_idx]
        ch_indices = self._ch_idx_t[valid_neuron_idx]

        gathered = channels[:, cell_indices, ch_indices]  # B, N_valid

        activation = torch.zeros(
            self._cell_idx_t.shape[0], vm.shape[0], device=self.device, dtype=self.dtype
        )
        activation[valid_neuron_idx, :] = gathered.transpose(0, 1).to(self.dtype)

        return activation

    def _build_mappings(self, root_ids: pd.DataFrame, tesselated_neurons: pd.DataFrame):
        num_neurons = len(root_ids)
        cell_idx = np.full(num_neurons, -1, dtype=np.int32)
        ch_idx = np.zeros(num_neurons, dtype=np.int8)

        tess = tesselated_neurons.copy()
        tess["root_id"] = tess["root_id"].astype("string")

        mapping_df = (
            root_ids.merge(
                tess[["root_id", "voronoi_indices", "cell_type"]],
                on="root_id",
                how="left",
            )
            .sort_values("index_id")
        )

        valid_mask = ~mapping_df["voronoi_indices"].isna()
        valid_indices = mapping_df[valid_mask]["index_id"].values.astype(int)
        cell_idx[valid_indices] = mapping_df[valid_mask]["voronoi_indices"].astype(int).values
        ch_idx[valid_indices] = mapping_df[valid_mask]["cell_type"].map(self._CHANNEL_MAP).astype(int).values

        # Torch tensors on device
        self._cell_idx_t = torch.tensor(cell_idx, device=self.device, dtype=torch.long)
        self._ch_idx_t = torch.tensor(ch_idx, device=self.device, dtype=torch.long)
        self._valid_neuron_idx = torch.nonzero(self._cell_idx_t != -1, as_tuple=False).squeeze()
