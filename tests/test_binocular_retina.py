import numpy as np
import pytest
import torch

from trainyourfly.config import Config
from trainyourfly.connectome_models.graph_builder import GraphBuilder
from trainyourfly.eye_models.binocular_retina import BinocularRetina
from trainyourfly.eye_models.neuron_mapper import NeuronMapper
from trainyourfly.eye_models.voronoi_cells import VoronoiCells
from trainyourfly.utils.csv_loader import CSVLoader

PIXEL_NUM = 32
DEVICE = torch.device("cpu")


@pytest.fixture
def builder(binocular_data_dir) -> GraphBuilder:
    config = Config(connectome_data_dir=binocular_data_dir, device_type="cpu")
    return GraphBuilder.from_dataset(
        data_dir=binocular_data_dir,
        csv_loader=CSVLoader(use_cache=False),
        rational_cell_types=["KCapbp-m"],
        config=config,
    )


@pytest.fixture
def retina(binocular_data_dir, builder) -> BinocularRetina:
    np.random.seed(3)
    return BinocularRetina(
        binocular_data_dir,
        builder.root_ids,
        pixel_num=PIXEL_NUM,
        device=DEVICE,
        dtype=torch.float32,
    )


def _photoreceptors(builder, side):
    return builder.node_indices_for_types(["R1-6", "R7", "R8"], side=side)


def test_components(retina):
    assert isinstance(retina.left, VoronoiCells)
    assert isinstance(retina.right, VoronoiCells)
    assert isinstance(retina.left_mapper, NeuronMapper)
    assert isinstance(retina.right_mapper, NeuronMapper)
    assert len(retina.left.centers) == 4
    assert len(retina.right.centers) == 4


def test_activations_shape_and_dtype(retina, builder):
    B = 3
    left = torch.rand(B, PIXEL_NUM, PIXEL_NUM, 3)
    right = torch.rand(B, PIXEL_NUM, PIXEL_NUM, 3)

    acts = retina.activations(left, right)

    assert acts.shape == (builder.num_nodes, B)
    assert acts.dtype == torch.float32
    assert acts.device.type == "cpu"
    assert torch.isfinite(acts).all()


def test_each_eye_drives_only_its_own_photoreceptors(retina, builder):
    white = torch.ones(1, PIXEL_NUM, PIXEL_NUM, 3)
    black = torch.zeros(1, PIXEL_NUM, PIXEL_NUM, 3)

    acts = retina.activations(white, black)[:, 0]

    left_idx = _photoreceptors(builder, "left")
    right_idx = _photoreceptors(builder, "right")
    assert len(left_idx) == 6 and len(right_idx) == 6

    # A uniformly white image gives 1 on every channel, so every left
    # photoreceptor reads 1 whichever channel it is tuned to
    assert torch.allclose(acts[left_idx], torch.ones(6))
    assert torch.equal(acts[right_idx], torch.zeros(6))

    central = np.setdiff1d(np.arange(builder.num_nodes), np.union1d(left_idx, right_idx))
    assert torch.equal(acts[central], torch.zeros(len(central)))


def test_activations_are_the_sum_of_both_eyes(retina):
    torch.manual_seed(11)
    left = torch.rand(2, PIXEL_NUM, PIXEL_NUM, 3)
    right = torch.rand(2, PIXEL_NUM, PIXEL_NUM, 3)
    black = torch.zeros_like(left)

    both = retina.activations(left, right)
    separate = retina.activations(left, black) + retina.activations(black, right)
    assert torch.allclose(both, separate)


def test_grayscale_equals_replicated_rgb(retina):
    torch.manual_seed(5)
    gray_left = torch.rand(2, PIXEL_NUM, PIXEL_NUM)
    gray_right = torch.rand(2, PIXEL_NUM, PIXEL_NUM)

    from_gray = retina.activations(gray_left, gray_right)
    from_rgb = retina.activations(
        gray_left.unsqueeze(-1).repeat(1, 1, 1, 3),
        gray_right.unsqueeze(-1).repeat(1, 1, 1, 3),
    )
    assert torch.allclose(from_gray, from_rgb)


def test_top_half_of_the_image_reaches_the_upper_ommatidia(retina, builder):
    # Image rows run top to bottom; neuron y_axis runs bottom to top, so the
    # R7 cells at y_axis = 400 look at the top half of the image
    left = torch.zeros(1, PIXEL_NUM, PIXEL_NUM, 3)
    left[:, : PIXEL_NUM // 2] = 1.0
    right = torch.zeros_like(left)

    acts = retina.activations(left, right)[:, 0]

    r7_idx = builder.node_indices_for_types(["R7"], side="left")
    root_ids = builder.root_ids.set_index("index_id")["root_id"]
    tess = retina.left.get_tesselated_neurons()
    y_axis_by_root = tess.set_index(tess["root_id"].astype(str))["y_axis"]
    for node in r7_idx:
        y_axis = y_axis_by_root[root_ids[node]]
        expected = 1.0 if y_axis > 256 else 0.0
        assert acts[node].item() == pytest.approx(expected)


def test_colour_channels_reach_the_right_photoreceptors(retina, builder):
    # Pure blue image: R7 (blue) and R1-6 (mean = 1/3) respond, R8 does not
    left = torch.zeros(1, PIXEL_NUM, PIXEL_NUM, 3)
    left[..., 2] = 1.0
    right = torch.zeros_like(left)

    acts = retina.activations(left, right)[:, 0]

    r7 = builder.node_indices_for_types(["R7"], side="left")
    r8 = builder.node_indices_for_types(["R8"], side="left")
    r16 = builder.node_indices_for_types(["R1-6"], side="left")
    assert torch.allclose(acts[r7], torch.ones(len(r7)))
    assert torch.equal(acts[r8], torch.zeros(len(r8)))
    assert torch.allclose(acts[r16], torch.full((len(r16),), 1 / 3))


def test_wrong_resolution_is_rejected(retina):
    imgs = torch.zeros(1, PIXEL_NUM + 1, PIXEL_NUM + 1, 3)
    with pytest.raises(AssertionError):
        retina.activations(imgs, imgs)


def test_random_voronoi_criteria_builds(binocular_data_dir, builder, monkeypatch):
    # Random seeds draw one centre per `ommatidia_size` neurons; the fixture
    # has six neurons per eye, so use every neuron as a seed
    monkeypatch.setattr(VoronoiCells, "ommatidia_size", 1)
    np.random.seed(3)
    retina = BinocularRetina(
        binocular_data_dir,
        builder.root_ids,
        pixel_num=PIXEL_NUM,
        device=DEVICE,
        dtype=torch.float32,
        voronoi_criteria="all",
    )
    acts = retina.activations(
        torch.rand(1, PIXEL_NUM, PIXEL_NUM, 3), torch.rand(1, PIXEL_NUM, PIXEL_NUM, 3)
    )
    assert acts.shape == (builder.num_nodes, 1)
    assert torch.isfinite(acts).all()
