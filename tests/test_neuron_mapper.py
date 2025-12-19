import pandas as pd
import torch
from trainyourfly.eye_models.neuron_mapper import NeuronMapper


def test_neuron_mapper_basic():
    # root_ids mapping
    root_ids = pd.DataFrame({"root_id": ["n1", "n2"], "index_id": [0, 1]})
    # tesselated neurons with cell types
    tess = pd.DataFrame({
        "root_id": ["n1", "n2"],
        "voronoi_indices": [0, 1],
        "cell_type": ["R7", "R8p"]
    })
    nm = NeuronMapper(root_ids, tess, device=torch.device("cpu"), dtype=torch.float32)

    # Build voronoi means tensor: B=1, cells=2
    # channels order r,g,b,mean
    vm = torch.tensor([[[0.1,0.2,0.9,0.4], [0.5,0.6,0.7,0.6]]])
    acts = nm.activations_from_voronoi_means(vm)
    # n1 is R7 -> b channel (index 2) of cell0 =0.9
    # n2 is R8p -> g channel (index1) of cell1 =0.6
    assert torch.allclose(acts[:,0], torch.tensor([0.9,0.6]))
