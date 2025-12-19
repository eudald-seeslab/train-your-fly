import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, cKDTree, voronoi_plot_2d
import torch
from torch_scatter import scatter_add

from trainyourfly.utils.train_funcs import assign_cell_type


class VoronoiCells:
    pixel_num = 512
    ommatidia_size = 8
    data_cols = ["x_axis", "y_axis"]
    centers = None
    voronoi = None
    tree = None
    index_map = None
    color_map = {"R1-6": "gray", "R7": "red", "R8p": "green", "R8y": "blue"}

    def __init__(self, data_dir, eye="right", neurons="all", voronoi_criteria="all"):
        self.data_dir = data_dir
        self.img_coords = self.get_image_coords(self.pixel_num)
        self.neuron_data = self._get_visual_neurons_data(neurons, eye)
        # if we use the R7 neurons as centers, we can create them already,
        # since they are fixed
        if voronoi_criteria == "R7":
            self.centers = self.get_R7_neurons(self.neuron_data)
            self.voronoi = Voronoi(self.centers)
            self.tree = cKDTree(self.centers)

    def query_points(self, points):
        _, indices = self.tree.query(points)

        return indices

    def get_tesselated_neurons(self):
        neuron_data = self.neuron_data.copy()

        neuron_indices = self.query_points(neuron_data[self.data_cols].values)

        neuron_data["voronoi_indices"] = neuron_indices
        neuron_data["cell_type"] = neuron_data.apply(assign_cell_type, axis=1)

        return neuron_data

    def get_image_indices(self):
        return self.query_points(self.img_coords)

    def _get_visual_neurons_data(self, neurons, side="right"):
        file = f"{side}_visual_positions_{neurons}_neurons.csv"
        data_path = os.path.join(self.data_dir, file)

        return pd.read_csv(data_path).drop(columns=["x", "y", "z", "PC1", "PC2"])

    def regenerate_random_centers(self):

        n_centers = self.neuron_data.shape[0] // self.ommatidia_size
        self.centers = self.neuron_data[self.data_cols].sample(n_centers).values
        self.voronoi = Voronoi(self.centers)
        self.tree = cKDTree(self.centers)

    def get_R7_neurons(self, neuron_data):
        return neuron_data[neuron_data["cell_type"] == "R7"][self.data_cols].values

    @staticmethod
    def get_image_coords(pixel_num):
        coords = (
            np.array(
                np.meshgrid(np.arange(pixel_num), np.arange(pixel_num), indexing="xy")
            )
            .reshape(2, -1)
            .T
        )

        # Invert "y" to start from the bottom, like with the neurons
        coords[:, 1] = pixel_num - 1 - coords[:, 1]
        return coords

    def _plot_voronoi_cells(
        self, ax, show_points=False, line_color="orange", line_width=1
    ):
        """Plot the Voronoi diagram, flipping the y-coordinate so that it
        matches the coordinate system used for both the underlying image
        (drawn with ``imshow`` and the default *origin='upper'*) and the
        neuron scatter plot, where the y-axis is already inverted
        Note that we create a *temporary* Voronoi instance whose centres have their
        y-coordinate flipped.  This way we do **not** touch
        ``self.voronoi`` (which is also used for KD-Tree queries etc.) and
        we avoid having to change the rest of the pipeline.
        """

        # Flip the points vertically so that they align with the neuron positions
        flipped_centres = self.centers.copy()
        flipped_centres[:, 1] = self.pixel_num - 1 - flipped_centres[:, 1]

        plot_voronoi = Voronoi(flipped_centres)

        voronoi_plot_2d(
            plot_voronoi,
            ax=ax,
            show_points=show_points,
            show_vertices=False,
            line_colors=line_color,
            line_width=line_width,
            line_alpha=0.8,
            point_size=2,
        )

    def plot_voronoi_cells_with_neurons(
        self, neuron_data, ax, voronoi_color, voronoi_width
    ):
        # Set black background
        ax.set_facecolor("black")

        # Modern color palette that pops on black:
        self.color_map = {
            "R1-6": "#ffffff",  # Bright mint
            "R7": "#1e90ff",  # Bright rose
            "R8p": "#50ff50",  # Bright gold
            "R8y": "#ff4d4d",  # Bright cyan
        }

        plot_data = neuron_data.copy()

        # The R7 and R8 neurons are in the exact same position, so we need
        # to jitter them to make them visible
        jitter_size = 4
        non_r1_6_mask = plot_data["cell_type"] != "R1-6"

        # Generate random choices between x and y jitter for each point
        x_choice = np.random.choice([-jitter_size, jitter_size], size=len(plot_data))

        # Where xy_choice is 0, make yx_choice ±2
        y_choice = np.random.choice([-jitter_size, jitter_size], size=len(plot_data))

        # Apply jitter only to non-R1-6 neurons
        plot_data.loc[non_r1_6_mask, "x_axis"] += x_choice[non_r1_6_mask]
        plot_data.loc[non_r1_6_mask, "y_axis"] += y_choice[non_r1_6_mask]

        plot_data["color"] = plot_data["cell_type"].apply(lambda x: self.color_map[x])

        self._plot_voronoi_cells(ax, line_color=voronoi_color, line_width=voronoi_width)

        plot_data["y_axis"] = self.pixel_num - 1 - plot_data["y_axis"]

        # Plot neurons with improved visibility
        for cell_type, color in self.color_map.items():
            points = plot_data[plot_data["cell_type"] == cell_type]
            ax.scatter(
                points["x_axis"],
                points["y_axis"],
                color=color,
                s=1 if cell_type == "R1-6" else 5,
                alpha=0.8 if cell_type == "R1-6" else 1,
                label=cell_type,
            )

        # Legend with white text
        legend = ax.legend(title="Cell Types", loc="lower right", frameon=True)
        legend.get_frame().set_facecolor("black")
        legend.get_frame().set_edgecolor("white")
        plt.setp(legend.get_texts(), color="white")
        plt.setp(legend.get_title(), color="white")

        self.clip_image(ax)

    def plot_voronoi_cells_with_image(self, image, ax):

        # Display the image
        ax.imshow(image, extent=[0, self.pixel_num, 0, self.pixel_num])

        # Plot Voronoi diagram
        self._plot_voronoi_cells(ax)
        self.clip_image(ax)

    def plot_input_image(self, image, ax):

        ax.imshow(image, extent=[0, self.pixel_num, 0, self.pixel_num])

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    def plot_neuron_activations(
        self, n_acts, ax, voronoi_color="orange", volonoi_width=1
    ):

        ax.set_facecolor("black")

        # Draw the Voronoi skeleton (with y-flip)
        self._plot_voronoi_cells(ax, line_color=voronoi_color, line_width=volonoi_width)

        rgb_values = (
            n_acts.loc[:, ["voronoi_indices", "cell_type", "activation"]]
            .groupby(["voronoi_indices", "cell_type"])
            .mean()
            .pivot_table(
                index="voronoi_indices", columns="cell_type", values="activation"
            )
            .fillna(0)
            .rename(columns={"R1-6": "mean", "R7": "b", "R8p": "g", "R8y": "r"})
        )

        rgb_values["b"] = self.get_colour_average(rgb_values, "b")
        rgb_values["g"] = self.get_colour_average(rgb_values, "g")
        rgb_values["r"] = self.get_colour_average(rgb_values, "r")

        # Build a *painting* Voronoi object whose geometry is flipped the
        # same way as in ``_plot_voronoi_cells`` so that filled polygons
        # coincide with the boundaries that were just drawn.
        flipped_centres = self.centers.copy()
        flipped_centres[:, 1] = self.pixel_num - 1 - flipped_centres[:, 1]
        plot_voronoi = Voronoi(flipped_centres)

        # Fill Voronoi regions: get colour from the *original* region index
        # (used to build ``rgb_values``) but draw the polygon determined by
        # the *flipped* geometry so that everything aligns.
        for i in range(len(self.centers)):
            region_orig = self.voronoi.point_region[i]
            region_flip = plot_voronoi.point_region[i]

            # Skip if the original region did not get an activation entry
            if region_orig not in rgb_values.index:
                continue

            region = plot_voronoi.regions[region_flip]
            if -1 in region:
                continue  # skip infinite regions

            polygon = [plot_voronoi.vertices[v] for v in region]
            color = rgb_values.loc[region_orig, ["r", "g", "b"]]
            ax.fill(*zip(*polygon), color=color)

        self.clip_image(ax)

    def clip_image(self, ax):
        # Set the axis limits to exactly 0 - pixel_num
        ax.set_xlim(0, self.pixel_num)
        ax.set_ylim(0, self.pixel_num)

        # Remove axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Ensure the aspect ratio is equal
        ax.set_aspect("equal", "box")

        # Optionally, you might want to set the edge color of the Voronoi cells to a light color
        # for better visibility against the black background
        ax.collections[-1].set_edgecolor("orange")
        ax.collections[-1].set_linewidth(0.5)

        # Tight layout to remove any extra white space
        # plt.tight_layout()

    @staticmethod
    def get_colour_average(values, colour):
        return (values[colour] + values["mean"]) / 2

    @staticmethod
    def compute_voronoi_means(
        processed_imgs: torch.Tensor,
        device: torch.device,
        pixel_counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-cell mean colour channels.

        Parameters
        ----------
        processed_imgs : Tensor
            Shape ``(B, P, 5)`` – ``[r, g, b, mean, cell_idx]``.
        device : torch.device
            Destination device for the computation.
        pixel_counts : Tensor | None, optional
            1-D tensor ``(C,)`` with the number of pixels belonging to each
            Voronoi cell.  If *None*, the counts are recomputed on the fly via
            ``scatter_add``.

        Returns
        -------
        Tensor
            Shape ``(B, C, 4)`` containing per-cell RGB/mean values.
        """
        processed = processed_imgs.to(device)
        B, P, _ = processed.shape
        channels = processed[:, :, :4]  # (B,P,4)
        cell_idx = processed[:, :, 4].long()  # (B,P)
        num_cells = int(cell_idx[0].max().item()) + 1

        # give each batch its own index range [0..num_cells-1] -> [k*num_cells ..]
        batch_offsets = torch.arange(B, device=device).view(B, 1) * num_cells
        flat_idx = (cell_idx + batch_offsets).reshape(-1)  # (B*P)
        flat_vals = channels.reshape(-1, 4)  # (B*P,4)

        total = B * num_cells
        sums = scatter_add(
            flat_vals,
            flat_idx.unsqueeze(-1).expand(-1, 4),
            dim=0,
            dim_size=total,
        )  # (total,4)

        # ------------------------------------------------------------
        # Counts: either use the supplied *pixel_counts* tensor or
        # compute them on the fly when *None* so that the function
        # behaves correctly with the default argument (needed by tests).
        # ------------------------------------------------------------
        if pixel_counts is None:
            # Compute per-cell pixel counts once and broadcast to all batches
            ones = torch.ones(P, device=device, dtype=channels.dtype)
            counts_single = scatter_add(
                ones, cell_idx[0], dim=0, dim_size=num_cells
            )  # (C,)
            counts = counts_single.repeat(B)  # (B*C,)
        else:
            counts = pixel_counts.to(device=device, dtype=channels.dtype).repeat(B)

        return (sums / counts.unsqueeze(-1)).view(B, num_cells, 4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recreate(self):
        """Generate a new random Voronoi tessellation and return the updated
        neuron mapping and image-pixel indices.

        Returns
        -------
        tuple (pd.DataFrame, np.ndarray)
            *Tesselated neurons* with updated ``voronoi_indices`` column and
            the flat array of per-pixel cell indices (length 512×512).
        """

        self.regenerate_random_centers()
        tess = self.get_tesselated_neurons()
        img_idx = self.get_image_indices()
        return tess, img_idx
