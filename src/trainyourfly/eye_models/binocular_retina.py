from __future__ import annotations

import pandas as pd
import torch

from trainyourfly.eye_models.neuron_mapper import NeuronMapper
from trainyourfly.eye_models.voronoi_cells import VoronoiCells


class BinocularRetina:
    """Two-eyed retina: one image per eye in, one activation per neuron out.

    Each eye is a :class:`VoronoiCells` tessellation seeded at that eye's R7
    terminals plus a :class:`NeuronMapper` that turns per-ommatidium colour
    averages into photoreceptor activations. Images are ``pixel_num`` pixels
    on a side and are mapped onto the 512 frame of the tessellation without
    resizing, so ``pixel_num`` can be much smaller than 512.

    The photoreceptors of the two eyes are disjoint sets of neurons, so the
    per-eye activation tensors are simply summed.

    Parameters
    ----------
    data_dir : str
        Folder with ``{left,right}_visual_positions_{neurons}_neurons.csv``.
    root_ids : pd.DataFrame
        ``root_id`` / ``index_id`` table of the graph (``GraphBuilder.root_ids``);
        the output is aligned with ``index_id``.
    pixel_num : int
        Side of the square input images.
    neurons : str
        ``"all"`` or ``"selected"``, selects the visual positions file.
    device, dtype : torch.device, torch.dtype
        Device and dtype of the returned activations.
    inhibitory_r7_r8 : bool
        Mutual inhibition between R7 and R8 inside an ommatidium.
    voronoi_criteria : str
        ``"R7"`` seeds the cells at the R7 terminals; anything else draws
        random seeds (``VoronoiCells.regenerate_random_centers``).
    """

    def __init__(
        self,
        data_dir: str,
        root_ids: pd.DataFrame,
        *,
        pixel_num: int,
        neurons: str = "all",
        device: torch.device,
        dtype: torch.dtype,
        inhibitory_r7_r8: bool = False,
        voronoi_criteria: str = "R7",
    ) -> None:
        self.pixel_num = pixel_num
        self.device = device
        self.dtype = dtype

        self.left = self._build_eye(data_dir, "left", neurons, voronoi_criteria)
        self.right = self._build_eye(data_dir, "right", neurons, voronoi_criteria)

        self.left_mapper = self._build_mapper(root_ids, self.left, inhibitory_r7_r8)
        self.right_mapper = self._build_mapper(root_ids, self.right, inhibitory_r7_r8)

        self._left_cell_idx, self._left_counts = self._pixel_cells(self.left)
        self._right_cell_idx, self._right_counts = self._pixel_cells(self.right)

    @staticmethod
    def _build_eye(data_dir, eye, neurons, voronoi_criteria) -> VoronoiCells:
        cells = VoronoiCells(
            data_dir, eye=eye, neurons=neurons, voronoi_criteria=voronoi_criteria
        )
        if voronoi_criteria != "R7":
            cells.regenerate_random_centers()
        return cells

    def _build_mapper(self, root_ids, cells, inhibitory_r7_r8) -> NeuronMapper:
        return NeuronMapper(
            root_ids,
            cells.get_tesselated_neurons(),
            device=self.device,
            dtype=self.dtype,
            inhibitory_r7_r8=inhibitory_r7_r8,
        )

    def _pixel_cells(self, cells: VoronoiCells) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-pixel cell index at ``pixel_num`` resolution and per-cell pixel
        counts. Cells that receive no pixel get a count of one so that their
        mean colour is zero instead of NaN."""
        cell_idx = torch.as_tensor(
            cells.get_image_indices(self.pixel_num), device=self.device
        ).long()
        counts = torch.bincount(cell_idx, minlength=len(cells.centers))
        counts = counts.clamp(min=1).to(dtype=torch.float32)
        return cell_idx, counts

    def activations(
        self, left_imgs: torch.Tensor, right_imgs: torch.Tensor
    ) -> torch.Tensor:
        """Photoreceptor activations for a batch of image pairs.

        Parameters
        ----------
        left_imgs, right_imgs : Tensor
            Float images in ``[0, 1]`` of shape ``(B, H, W, 3)`` or ``(B, H, W)``
            (grayscale, replicated to three channels), with
            ``H == W == pixel_num``.

        Returns
        -------
        Tensor
            Shape ``(num_nodes, B)``; zero for every neuron that is not a
            photoreceptor.
        """
        left = self._eye_activations(
            left_imgs, self._left_cell_idx, self._left_counts, self.left_mapper
        )
        right = self._eye_activations(
            right_imgs, self._right_cell_idx, self._right_counts, self.right_mapper
        )
        return left + right

    def _eye_activations(
        self,
        imgs: torch.Tensor,
        cell_idx: torch.Tensor,
        counts: torch.Tensor,
        mapper: NeuronMapper,
    ) -> torch.Tensor:
        imgs = torch.as_tensor(imgs).to(device=self.device, dtype=torch.float32)
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(-1).expand(-1, -1, -1, 3)

        B, H, W, C = imgs.shape
        assert H == W == self.pixel_num, (
            f"images must be {self.pixel_num} x {self.pixel_num}, got {H} x {W}"
        )
        assert C == 3, f"images must have 3 channels, got {C}"

        # (B, P, 5): [r, g, b, mean, cell_idx]
        flat = imgs.reshape(B, -1, 3)
        mean = flat.mean(dim=2, keepdim=True)
        idx = cell_idx.view(1, -1, 1).expand(B, -1, 1).to(flat.dtype)
        processed = torch.cat([flat, mean, idx], dim=2)

        means = VoronoiCells.compute_voronoi_means(processed, self.device, counts)
        return mapper.activations_from_voronoi_means(means)
