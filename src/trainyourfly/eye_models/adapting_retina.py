from __future__ import annotations

import torch

from trainyourfly.eye_models.binocular_retina import BinocularRetina


class AdaptingRetina:
    """A :class:`BinocularRetina` whose photoreceptors adapt to the light.

    Every ommatidium keeps, per channel, a slow running mean of the light it
    has been collecting (``adapted``), and its photoreceptors report the light
    they get now relative to it::

        activation = light - adapted
        adapted   += rate * (light - adapted)

    What an eye that looks around has got used to is the scene averaged over
    where it has been looking, so a uniform background, however bright,
    reports nothing, and what stands out of it (an object, an edge, something
    that moves or appears) reports its contrast, with its sign: brighter or
    darker than the eye expected. A scene that does not change fades in about
    ``1 / rate`` steps. Fly photoreceptors and the lamina cells they drive
    respond to contrast in this way.

    The state is one adapted level per image pair of the batch, so the batch
    is a fixed population of eyes seen over time, not a set of unrelated
    images. An eye starts adapted to the first thing it sees (see
    :meth:`reset`): opening the eyes is not a flash.

    Parameters
    ----------
    retina : BinocularRetina
        The eyes; images, devices and the returned activations are its own.
    batch_size : int
        Number of eye pairs.
    rate : float
        Share of the distance to the current light that the adapted level
        covers per step, in ``(0, 1]``.
    """

    def __init__(self, retina: BinocularRetina, batch_size: int, rate: float) -> None:
        if not 0.0 < rate <= 1.0:
            raise ValueError(f"rate must be in (0, 1], got {rate}")
        self.retina = retina
        self.rate = rate
        self.adapted_left = torch.zeros(
            batch_size, len(retina.left.centers), 4, device=retina.device
        )
        self.adapted_right = torch.zeros(
            batch_size, len(retina.right.centers), 4, device=retina.device
        )
        self.unadapted = torch.ones(batch_size, dtype=torch.bool, device=retina.device)
        """Eyes that have not seen anything yet: the next step sets their level."""

    def reset(self, index: torch.Tensor | None = None) -> None:
        """Forget what the eyes ``index`` (all of them when ``None``) were
        adapted to; they adapt at once to the next thing they see."""
        if index is None:
            index = torch.arange(self.unadapted.shape[0], device=self.unadapted.device)
        self.adapted_left[index] = 0.0
        self.adapted_right[index] = 0.0
        self.unadapted[index] = True

    def activations(
        self, left_imgs: torch.Tensor, right_imgs: torch.Tensor
    ) -> torch.Tensor:
        """Adapted photoreceptor activations ``(num_nodes, B)`` for the next
        image pair of every eye, with ``B == batch_size``; images as in
        :meth:`BinocularRetina.activations`. Advances the adapted levels."""
        left, right = self.retina.cell_means(left_imgs, right_imgs)
        if bool(self.unadapted.any()):
            self.adapted_left[self.unadapted] = left[self.unadapted]
            self.adapted_right[self.unadapted] = right[self.unadapted]
            self.unadapted[:] = False
        out = self.retina.activations_from_cell_means(
            left - self.adapted_left, right - self.adapted_right
        )
        self.adapted_left.lerp_(left, self.rate)
        self.adapted_right.lerp_(right, self.rate)
        return out

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "adapted_left": self.adapted_left.detach().cpu().clone(),
            "adapted_right": self.adapted_right.detach().cpu().clone(),
            "unadapted": self.unadapted.detach().cpu().clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name in ("adapted_left", "adapted_right", "unadapted"):
            dst = getattr(self, name)
            dst.copy_(state[name].to(device=dst.device, dtype=dst.dtype))
