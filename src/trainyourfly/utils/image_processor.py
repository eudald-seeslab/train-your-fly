from __future__ import annotations

from typing import Union

import numpy as np
import torch
from torch.nn import functional as F


class ImageProcessor:
    """GPU-accelerated helper for image resizing and colour-channel stacking.

    Identical implementation to the previous *ImagePreprocessor* but renamed for
    brevity / clarity.  Kept in a separate module so we can later delete the
    compatibility alias.
    """

    def __init__(self, device: torch.device):
        self.device = device

    def preprocess(self, imgs_np: np.ndarray) -> torch.Tensor:
        """Resize grayscale images and replicate channels if needed (float32)."""
        return self._prepare_images(imgs_np, dtype=torch.float32, scale=False)

    def process(self, imgs_input: Union[np.ndarray, torch.Tensor], voronoi_indices_t: torch.Tensor) -> torch.Tensor:
        """Full pipeline: resize + /255 scaling + mean channel + Voronoi index."""
        target_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        imgs_t = self._prepare_images(imgs_input, dtype=target_dtype, scale=True)

        B = imgs_t.shape[0]

        imgs_t = imgs_t.reshape(B, -1, 3)  # flatten spatial
        mean_ch = imgs_t.mean(dim=2, keepdim=True)
        imgs_t = torch.cat([imgs_t, mean_ch], dim=2)

        vor_idx = voronoi_indices_t.view(1, -1, 1).expand(B, -1, 1).to(target_dtype)
        
        return torch.cat([imgs_t, vor_idx], dim=2)

    def _prepare_images(self, imgs: Union[np.ndarray, torch.Tensor], *, dtype: torch.dtype, scale: bool) -> torch.Tensor:
        """Common preprocessing sub-routine used by *preprocess* and *process*.

        Parameters
        ----------
        imgs : ndarray | Tensor
            Input images of shape (B,H,W) or (B,H,W,3).
        dtype : torch.dtype
            Target dtype of the returned tensor.
        scale : bool
            If *True*, divide by 255 to bring the range into [0,1].
        """

        imgs_t = torch.from_numpy(imgs) if isinstance(imgs, np.ndarray) else imgs
        imgs_t = imgs_t.to(self.device, dtype=dtype)

        # Grayscale → 3-channel RGB
        if imgs_t.ndim == 3:
            imgs_t = imgs_t.unsqueeze(-1).repeat(1, 1, 1, 3)

        # Resize to 512×512 if needed
        _, H, W, C = imgs_t.shape
        assert C == 3, "Images must have 3 channels"

        if H != 512 or W != 512:
            imgs_t = imgs_t.permute(0, 3, 1, 2).float()
            imgs_t = F.interpolate(imgs_t, size=(512, 512), mode="bilinear", align_corners=False)
            imgs_t = imgs_t.permute(0, 2, 3, 1)

        if scale:
            imgs_t = imgs_t / 255.0

        return imgs_t 