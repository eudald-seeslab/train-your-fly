import torch
from connectome.core.voronoi_cells import VoronoiCells


def test_compute_voronoi_means():
    # Minimal example: two pixels, two cells
    # Build processed_imgs tensor (B=1,P=2,5)
    # pixel0 belongs to cell0, pixel1 to cell1
    processed = torch.tensor([[[1.0,0.0,0.0,1.0,0], [0.0,1.0,0.0,1.0,1]]])
    means = VoronoiCells.compute_voronoi_means(processed, device=torch.device("cpu"))
    assert means.shape == (1,2,4)
    # r channel cell0 should be 1, g channel cell1 should be1
    assert torch.allclose(means[0,0,0], torch.tensor(1.0))
    assert torch.allclose(means[0,1,1], torch.tensor(1.0))
