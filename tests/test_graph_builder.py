import torch
from connectome.core.graph_builder import GraphBuilder

def test_graph_builder_basic():
    # two-node directed edge 0->1,1->0
    edges = torch.tensor([[0,1],[1,0]], dtype=torch.int32)
    gb = GraphBuilder(edges, device=torch.device("cpu"))

    activ = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (N_nodes=2, B=2)
    inputs, labels = gb.build_batch(activ, labels=[0, 1])

    # Node features should be flatten(batch,node)
    assert inputs.x.squeeze(1).tolist() == [1.0, 3.0, 2.0, 4.0]
    # Edge index should have offsets 0 and 2
    assert inputs.edge_index[:, :2].tolist() == [[0, 1], [1, 0]]
    assert inputs.edge_index[:, 2:].tolist() == [[2, 3], [3, 2]]
    # Batch vector 0,0,1,1
    assert inputs.batch.tolist() == [0, 0, 1, 1]
    assert labels.tolist() == [0, 1] 
