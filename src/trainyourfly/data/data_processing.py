import os
import random

import numpy as np
import torch

from paths import PROJECT_ROOT
from connectome.core.csv_loader import CSVLoader
from connectome.core.fly_plotter import FlyPlotter

from connectome.core.train_funcs import import_images
from connectome.core.voronoi_cells import VoronoiCells
from connectome.core.neuron_mapper import NeuronMapper
from connectome.core.utils import paths_to_labels
from connectome.core.image_processor import ImageProcessor
from connectome.core.graph_builder import GraphBuilder
from configs.config import CONNECTOME_DATA_DIR


class DataProcessor:

    tesselated_neurons = None
    retinal_cells = ["R1-6", "R7", "R8"]

    def __init__(self, config_, input_images_dir=None):
        """Initialise a ``DataProcessor``."""

        self._cfg = config_
        self._input_images_dir = input_images_dir

        # Step-wise initialisation
        # ------------------------------------------------------------------
        # 1. Reproducibility
        # ------------------------------------------------------------------
        self._init_randomness(config_.random_seed)

        # ------------------------------------------------------------------
        # 2. IO helpers and data paths
        # ------------------------------------------------------------------
        self.data_dir = os.path.join(PROJECT_ROOT, CONNECTOME_DATA_DIR)
        self.csv_loader = CSVLoader()

        # ------------------------------------------------------------------
        # 3. Neuron & synapse data
        # ------------------------------------------------------------------
        rational_cell_types = self.get_rational_cell_types(config_.rational_cell_types)
        self.protected_cell_types = self.retinal_cells + rational_cell_types
        self._check_filtered_neurons(config_.filtered_celltypes)

        # ------------------------------------------------------------------
        # Graph + synaptic data
        # ------------------------------------------------------------------
        self.graph_builder = GraphBuilder.from_dataset(
            data_dir=self.data_dir,
            csv_loader=self.csv_loader,
            rational_cell_types=rational_cell_types,
            config=config_,
        )

        # TODO: this should stay in graph_builder
        self.root_ids = self.graph_builder.root_ids

        # ------------------------------------------------------------------
        # 4. Voronoi geometry
        # ------------------------------------------------------------------
        self.voronoi_cells = VoronoiCells(
            data_dir=self.data_dir,
            eye=config_.eye,
            neurons=config_.neurons,
            voronoi_criteria=config_.voronoi_criteria,
        )

        if config_.voronoi_criteria == "R7":
            self.tesselated_neurons = self.voronoi_cells.get_tesselated_neurons()
            self.voronoi_indices = self.voronoi_cells.get_image_indices()
            self.voronoi_indices_torch = torch.tensor(
                self.voronoi_indices,
                device=config_.DEVICE,
                dtype=torch.long,
            )

            # Cache per-cell pixel counts once (small tensor, few kB)
            num_cells = int(self.voronoi_indices_torch.max().item()) + 1
            counts = torch.bincount(
                self.voronoi_indices_torch.cpu(), minlength=num_cells
            ).to(config_.DEVICE, dtype=torch.float32)
            self.pixel_counts_torch = counts
        else:
            self.tesselated_neurons = None
            self.voronoi_indices = None
            self.voronoi_indices_torch = None
            self.pixel_counts_torch = None

        # ------------------------------------------------------------------
        # 5. Device, dtypes & image helpers
        # ------------------------------------------------------------------
        self.dtype = config_.dtype
        self.device = config_.DEVICE
        self._image_processor = ImageProcessor(self.device)

        self.classes = (
            sorted(os.listdir(input_images_dir))
            if input_images_dir is not None
            else config_.CLASSES
        )

        # ------------------------------------------------------------------
        # 6. Graph representation helpers
        # ------------------------------------------------------------------
        self.inhibitory_r7_r8 = config_.inhibitory_r7_r8

        self.neuron_mapper = NeuronMapper(
            self.root_ids,
            self.tesselated_neurons,
            device=self.device,
            dtype=self.dtype,
            inhibitory_r7_r8=self.inhibitory_r7_r8,
        )

        # ------------------------------------------------------------------
        # 7. Plotting
        # ------------------------------------------------------------------
        self.plotter = FlyPlotter(self.voronoi_cells)

    @property
    def num_classes(self):
        return len(self.classes)

    def process_batch(self, imgs, labels):
        """
        Preprocesses a batch of images and labels. This includes reshaping and colouring the images if necessary,
        tesselating it according to the voronoi cells from the connectome, and getting the neuron activations for each
        cell. Finally, it constructs the graphs of this batch with the appropriate activations.

        Args:
            imgs (list): A list of images to be processed.
            labels (list): A list of corresponding labels.

        Returns:
            inputs (Batch): A Batch object containing processed graph data.
            labels (torch.Tensor): A tensor containing the labels.

        Raises:
            None

        """

        # Reshape and colour images if needed
        imgs_t = self._image_processor.preprocess(imgs)
        processed_imgs = self._image_processor.process(imgs_t, self.voronoi_indices_torch)
        voronoi_means = VoronoiCells.compute_voronoi_means(
            processed_imgs,
            self.device,
            self.pixel_counts_torch,
        )
        activation_tensor = self.neuron_mapper.activations_from_voronoi_means(voronoi_means)

        # Delete bulky intermediate tensors to reclaim GPU memory before constructing
        # the (potentially huge) batched edge index. 
        del imgs_t, processed_imgs, voronoi_means
        torch.cuda.empty_cache()

        # Convert activations into a batched PyG graph
        inputs, labels_t = self.graph_builder.build_batch(activation_tensor, labels)

        return inputs, labels_t

    def update_voronoi_state(self):
        """Refresh cached tensors after ``voronoi_cells.recreate()``."""

        # TODO: I'm not convinced about this method here. Think about it.
        self.tesselated_neurons = self.voronoi_cells.get_tesselated_neurons()
        self.voronoi_indices = self.voronoi_cells.get_image_indices()
        self.voronoi_indices_torch = torch.tensor(
            self.voronoi_indices, device=self.device, dtype=torch.long
        )

        self.neuron_mapper = NeuronMapper(
            self.root_ids,
            self.tesselated_neurons,
            device=self.device,
            dtype=self.dtype,
            inhibitory_r7_r8=self.inhibitory_r7_r8,
        )

        # Update cached pixel counts after the tessellation changes
        num_cells = int(self.voronoi_indices_torch.max().item()) + 1
        self.pixel_counts_torch = torch.bincount(
            self.voronoi_indices_torch.cpu(), minlength=num_cells
        ).to(self.device, dtype=torch.float32)

    def get_data_from_paths(self, paths, get_labels=True):
        imgs = import_images(paths)
        if get_labels:
            labels = paths_to_labels(paths, self.classes)
        else:
            labels = None
        return imgs, labels

    def plot_input_images(self, img, voronoi_colour="orange", voronoi_width=1):
        """Convenience wrapper that delegates to ConnectomePlotter."""
        return self.plotter.plot_input_images(
            img,
            self.tesselated_neurons,
            self.voronoi_indices,
            self.inhibitory_r7_r8,
            voronoi_colour=voronoi_colour,
            voronoi_width=voronoi_width,
        )

    def get_rational_cell_types_from_file(self):
        path = os.path.join(self.data_dir, "rational_cell_types.csv")
        df = self.csv_loader.read_csv(path, index_col=0)
        return df.index.tolist()

    def get_rational_cell_types(self, rational_cell_types):
        # TODO: not sure this should be here.
        if rational_cell_types is None:
            return self.get_rational_cell_types_from_file()
        return rational_cell_types

    def _check_filtered_neurons(self, filtered_cell_types):
        if not set(filtered_cell_types).isdisjoint(self.protected_cell_types):
            raise ValueError(
                f"You can't filter out any of the following cell types: {', '.join(self.protected_cell_types)}"
            )

    @staticmethod
    def _init_randomness(random_seed):
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)
