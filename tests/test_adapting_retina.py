import numpy as np
import pytest
import torch

from trainyourfly.config import Config
from trainyourfly.connectome_models.graph_builder import GraphBuilder
from trainyourfly.eye_models.adapting_retina import AdaptingRetina
from trainyourfly.eye_models.binocular_retina import BinocularRetina
from trainyourfly.utils.csv_loader import CSVLoader

PIXEL_NUM = 32
RATE = 0.25


@pytest.fixture
def retina(binocular_data_dir) -> BinocularRetina:
    config = Config(connectome_data_dir=binocular_data_dir, device_type="cpu")
    builder = GraphBuilder.from_dataset(
        data_dir=binocular_data_dir,
        csv_loader=CSVLoader(use_cache=False),
        rational_cell_types=["KCapbp-m"],
        config=config,
    )
    np.random.seed(3)
    return BinocularRetina(
        binocular_data_dir,
        builder.root_ids,
        pixel_num=PIXEL_NUM,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _images(batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return (
        torch.rand(batch, PIXEL_NUM, PIXEL_NUM, 3, generator=gen),
        torch.rand(batch, PIXEL_NUM, PIXEL_NUM, 3, generator=gen),
    )


def test_cell_means_then_mapping_is_activations(retina):
    left, right = _images(3, 0)
    means = retina.cell_means(left, right)
    assert means[0].shape == (3, len(retina.left.centers), 4)
    assert torch.equal(
        retina.activations_from_cell_means(*means), retina.activations(left, right)
    )


def test_the_first_sight_and_a_steady_scene_report_nothing(retina):
    eyes = AdaptingRetina(retina, 2, RATE)
    left, right = _images(2, 1)
    for _ in range(3):
        assert torch.equal(eyes.activations(left, right), torch.zeros_like(retina.activations(left, right)))


def test_a_change_reports_its_contrast_and_fades(retina):
    eyes = AdaptingRetina(retina, 2, RATE)
    before = _images(2, 2)
    after = _images(2, 3)
    eyes.activations(*before)

    change = retina.activations(*after) - retina.activations(*before)
    assert change.abs().max() > 0.01  # brighter in some ommatidia, darker in others
    for step in range(4):
        out = eyes.activations(*after)
        assert torch.allclose(out, change * (1 - RATE) ** step, atol=1e-6)


def test_a_uniform_background_is_invisible_whatever_its_brightness(retina):
    # Two eyes look at the same small bright patch, one on a dark background and
    # one on a bright one: once adapted to the background, both report the patch
    # with the same contrast.
    eyes = AdaptingRetina(retina, 2, 1.0)
    background = torch.zeros(2, PIXEL_NUM, PIXEL_NUM, 3)
    background[1] = 0.6
    eyes.activations(background, background)

    patch = background.clone()
    patch[:, 4:10, 4:10] += 0.3
    out = eyes.activations(patch, patch)
    assert out.abs().max() > 0.01
    assert torch.allclose(out[:, 0], out[:, 1], atol=1e-6)


def test_reset_concerns_the_given_eyes_only(retina):
    eyes = AdaptingRetina(retina, 2, RATE)
    before = _images(2, 4)
    after = _images(2, 5)
    eyes.activations(*before)
    eyes.reset(torch.tensor([1]))

    out = eyes.activations(*after)
    change = retina.activations(*after) - retina.activations(*before)
    assert torch.allclose(out[:, 0], change[:, 0], atol=1e-6)
    assert torch.equal(out[:, 1], torch.zeros_like(out[:, 1]))


def test_state_round_trip(retina):
    eyes = AdaptingRetina(retina, 2, RATE)
    eyes.activations(*_images(2, 6))
    other = AdaptingRetina(retina, 2, RATE)
    other.load_state_dict(eyes.state_dict())
    nxt = _images(2, 7)
    assert torch.equal(eyes.activations(*nxt), other.activations(*nxt))


def test_rate_must_be_a_share(retina):
    with pytest.raises(ValueError):
        AdaptingRetina(retina, 2, 0.0)
