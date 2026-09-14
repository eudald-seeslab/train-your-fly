import numpy as np
import pytest
import torch

from trainyourfly.eye_models.voronoi_cells import VoronoiCells


@pytest.fixture
def cells(binocular_data_dir) -> VoronoiCells:
    return VoronoiCells(binocular_data_dir, eye="right", voronoi_criteria="R7")


def test_get_image_coords_native_grid_unchanged():
    coords = VoronoiCells.get_image_coords(4)
    assert coords.shape == (16, 2)
    # Row-major: first pixel is the top-left corner, y counted from the bottom
    assert coords[0].tolist() == [0, 3]
    assert coords[3].tolist() == [3, 3]
    assert coords[-1].tolist() == [3, 0]
    assert np.array_equal(
        VoronoiCells.get_image_coords(4, frame_size=4), coords.astype(float)
    )


def test_get_image_coords_scaled_to_frame():
    coords = VoronoiCells.get_image_coords(4, frame_size=512)
    # Each low-res pixel sits at the centre of the 128-pixel block it covers
    assert coords[0].tolist() == [63.5, 511 - 63.5]
    assert coords[-1].tolist() == [511 - 63.5, 63.5]
    assert coords.min() >= 0 and coords.max() <= 511


def test_get_image_indices_default_is_native_512(cells):
    default = cells.get_image_indices()
    explicit = cells.get_image_indices(512)
    assert default.shape == (512 * 512,)
    assert np.array_equal(default, explicit)
    assert np.array_equal(default, cells.query_points(cells.img_coords))


def test_get_image_indices_low_resolution_matches_brute_force(cells):
    pixel_num = 32
    indices = cells.get_image_indices(pixel_num)
    assert indices.shape == (pixel_num**2,)

    coords = VoronoiCells.get_image_coords(pixel_num, frame_size=512)
    distances = np.linalg.norm(coords[:, None, :] - cells.centers[None, :, :], axis=2)
    assert np.array_equal(indices, distances.argmin(axis=1))

    # With centres at the corners of a square, every cell covers one quadrant
    counts = np.bincount(indices, minlength=len(cells.centers))
    assert np.all(counts == pixel_num**2 // 4)


def test_low_resolution_quadrants_agree_with_native_grid(cells):
    pixel_num = 16
    low = cells.get_image_indices(pixel_num).reshape(pixel_num, pixel_num)
    native = cells.get_image_indices().reshape(512, 512)

    # Centre of each low-res pixel block, in native image coordinates
    block = 512 // pixel_num
    rows = np.arange(pixel_num) * block + block // 2
    assert np.array_equal(low, native[np.ix_(rows, rows)])


def test_compute_voronoi_means_keeps_empty_cells_when_counts_given():
    # Two pixels both in cell 0; pixel_counts declares three cells
    processed = torch.tensor([[[1.0, 0.5, 0.0, 0.5, 0], [0.0, 0.5, 1.0, 0.5, 0]]])
    counts = torch.tensor([2.0, 1.0, 1.0])
    means = VoronoiCells.compute_voronoi_means(processed, torch.device("cpu"), counts)

    assert means.shape == (1, 3, 4)
    assert torch.allclose(means[0, 0], torch.tensor([0.5, 0.5, 0.5, 0.5]))
    assert torch.equal(means[0, 1:], torch.zeros(2, 4))
