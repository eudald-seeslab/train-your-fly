import numpy as np
from matplotlib import pyplot as plt
import torch

from connectome.core.train_funcs import (
    apply_inhibitory_r7_r8,
    get_activation_from_cell_type,
    get_voronoi_averages,
    preprocess_images,
    process_images,
)


class FlyPlotter:
    """Utility class for generating diagnostic plots during training."""

    def __init__(self, voronoi_cells):
        self.voronoi_cells = voronoi_cells

    def plot_input_images(self, img, tesselated_neurons, voronoi_indices, inhibitory_r7_r8=False, *, voronoi_colour="orange", voronoi_width=1):
        """Return (figure, title) tuple with three-panel diagnostic plot."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        self._plot_voronoi_cells_with_neurons(tesselated_neurons, axes[0], voronoi_colour, voronoi_width)
        self._plot_neuron_activations(img, axes[1], tesselated_neurons, voronoi_indices, inhibitory_r7_r8, voronoi_colour, voronoi_width)
        self._plot_input_image(img, axes[2])
        plt.tight_layout()
        plt.close("all")
        return fig, "Activations -> Voronoi <- Input image"

    def _plot_voronoi_cells_with_neurons(self, neuron_data, ax, voronoi_color, voronoi_width):
        vc = self.voronoi_cells
        # Set black background
        ax.set_facecolor("black")

        color_map = {
            "R1-6": "#ffffff",
            "R7": "#1e90ff",
            "R8p": "#50ff50",
            "R8y": "#ff4d4d",
        }

        plot_data = neuron_data.copy()
        jitter_size = 4
        non_r1_6_mask = plot_data["cell_type"] != "R1-6"
        x_choice = np.random.choice([-jitter_size, jitter_size], size=len(plot_data))
        y_choice = np.random.choice([-jitter_size, jitter_size], size=len(plot_data))
        plot_data.loc[non_r1_6_mask, "x_axis"] += x_choice[non_r1_6_mask]
        plot_data.loc[non_r1_6_mask, "y_axis"] += y_choice[non_r1_6_mask]
        plot_data["color"] = plot_data["cell_type"].apply(lambda x: color_map[x])

        self._plot_voronoi_skeleton(ax, voronoi_color, voronoi_width)
        plot_data["y_axis"] = vc.pixel_num - 1 - plot_data["y_axis"]
        for cell_type, color in color_map.items():
            points = plot_data[plot_data["cell_type"] == cell_type]
            ax.scatter(
                points["x_axis"],
                points["y_axis"],
                color=color,
                s=1 if cell_type == "R1-6" else 5,
                alpha=0.8 if cell_type == "R1-6" else 1,
                label=cell_type,
            )

        legend = ax.legend(title="Cell Types", loc="lower right", frameon=True)
        legend.get_frame().set_facecolor("black")
        legend.get_frame().set_edgecolor("white")
        for text in legend.get_texts():
            text.set_color("white")
        legend.get_title().set_color("white")
        self._clip_image(ax)

    def _plot_neuron_activations(self, img, ax, tesselated_neurons, voronoi_indices, inhibitory_r7_r8, voronoi_color, voronoi_width):
        vc = self.voronoi_cells
        img_pre = preprocess_images(np.expand_dims(img, 0))
        processed_img = process_images(img_pre, voronoi_indices)
        voronoi_average = get_voronoi_averages(processed_img)[0]
        voronoi_average.index = [vc.voronoi.point_region[int(i)] for i in voronoi_average.index]
        corr_tess = tesselated_neurons.copy()
        corr_tess["voronoi_indices"] = [vc.voronoi.point_region[int(i)] for i in corr_tess["voronoi_indices"]]
        neuron_acts = corr_tess.merge(voronoi_average, left_on="voronoi_indices", right_index=True)
        if inhibitory_r7_r8:
            neuron_acts = apply_inhibitory_r7_r8(neuron_acts)
        neuron_acts["activation"] = neuron_acts.apply(get_activation_from_cell_type, axis=1)
        self._plot_voronoi_neuron_activations(neuron_acts, ax, voronoi_color, voronoi_width)

    def _plot_voronoi_skeleton(self, ax, line_color, line_width):
        vc = self.voronoi_cells
        centres = vc.centers.copy()
        centres[:, 1] = vc.pixel_num - 1 - centres[:, 1]
        from scipy.spatial import Voronoi, voronoi_plot_2d
        plot_voronoi = Voronoi(centres)
        voronoi_plot_2d(plot_voronoi, ax=ax, show_points=False, show_vertices=False, line_colors=line_color, line_width=line_width, line_alpha=0.8, point_size=2)

    def _plot_input_image(self, image, ax):
        vc = self.voronoi_cells
        # Use origin='lower' so that the image coordinate system matches the
        # one used for the Voronoi geometry (y increases upwards).
        ax.imshow(image, extent=[0, vc.pixel_num, 0, vc.pixel_num], origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])

    def _plot_voronoi_neuron_activations(self, n_acts, ax, voronoi_color, voronoi_width):
        vc = self.voronoi_cells
        ax.set_facecolor("black")
        self._plot_voronoi_skeleton(ax, voronoi_color, voronoi_width)
        rgb_values = (
            n_acts.loc[:, ["voronoi_indices", "cell_type", "activation"]]
            .groupby(["voronoi_indices", "cell_type"]).mean()
            .pivot_table(index="voronoi_indices", columns="cell_type", values="activation")
            .fillna(0)
            .rename(columns={"R1-6": "mean", "R7": "b", "R8p": "g", "R8y": "r"})
        )
        rgb_values["b"] = (rgb_values["b"] + rgb_values["mean"]) / 2
        rgb_values["g"] = (rgb_values["g"] + rgb_values["mean"]) / 2
        rgb_values["r"] = (rgb_values["r"] + rgb_values["mean"]) / 2
        centres = vc.centers.copy()
        centres[:, 1] = vc.pixel_num - 1 - centres[:, 1]
        from scipy.spatial import Voronoi
        plot_voronoi = Voronoi(centres)
        for i in range(len(centres)):
            region_orig = vc.voronoi.point_region[i]
            region_flip = plot_voronoi.point_region[i]
            if region_orig not in rgb_values.index:
                continue
            region = plot_voronoi.regions[region_flip]
            if -1 in region:
                continue
            polygon = [plot_voronoi.vertices[v] for v in region]
            color = rgb_values.loc[region_orig, ["r", "g", "b"]]
            ax.fill(*zip(*polygon), color=color)
        self._clip_image(ax)

    def _clip_image(self, ax):
        vc = self.voronoi_cells
        ax.set_xlim(0, vc.pixel_num)
        ax.set_ylim(0, vc.pixel_num)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", "box")
