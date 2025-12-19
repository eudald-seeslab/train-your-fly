import numpy as np
import pandas as pd
from trainyourfly.eye_models.voronoi_cells import VoronoiCells
from trainyourfly.plots.fly_plotter import FlyPlotter
import matplotlib
from types import MethodType

matplotlib.use("Agg")

def test_fly_plotter_runs():
    # minimal VoronoiCells stub
    vc = VoronoiCells.__new__(VoronoiCells)
    vc.pixel_num = 512
    vc.centers = np.array([[2, 2], [5, 5], [2, 5]])
    from scipy.spatial import Voronoi
    vc.voronoi = Voronoi(vc.centers)
    vc.tree = None

    fp = FlyPlotter(vc)

    img = np.zeros((512, 512, 3), dtype=np.uint8)

    # minimal neuron dataframe with required columns
    neuron_df = pd.DataFrame({
        "x_axis": [10, 20],
        "y_axis": [10, 20],
        "cell_type": ["R7", "R8p"],
        "voronoi_indices": [0, 1],
    })

    # Monkey-patch the heavy internal call so we only test wiring/figure creation
    fp._plot_neuron_activations = MethodType(lambda self, *a, **k: None, fp)

    try:
        fp.plot_input_images(img, neuron_df, np.zeros(vc.pixel_num * vc.pixel_num, dtype=int))
    except Exception:
        assert False, "FlyPlotter raised unexpectedly" 
