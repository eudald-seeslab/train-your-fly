import torch
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from trainyourfly.connectome_models.graph_models_helpers import log_norm, min_max_norm


class Connectome(MessagePassing):
    def __init__(self, data_processor, config):
        super(Connectome, self).__init__(aggr="add")

        self.num_passes = config.NUM_CONNECTOME_PASSES
        graph_builder = data_processor.graph_builder
        num_nodes = graph_builder.num_nodes
        edge_weight = graph_builder.synaptic_matrix.data
        num_synapses = edge_weight.shape[0]
        self.batch_size = config.batch_size
        self.train_edges = config.train_edges
        self.train_neurons = config.train_neurons
        self.lambda_func = config.lambda_func
        self.neuron_normalization = config.neuron_normalization
        self.refined_synaptic_data = config.refined_synaptic_data
        self.synaptic_limit = config.synaptic_limit
        dtype = config.dtype
        device = config.DEVICE

        self.register_buffer(
            "edge_weight",
            torch.tensor(edge_weight, dtype=dtype, device=device),
        )

        if config.train_edges:
            # This allows us to have negative weights, as well as synapses comprised
            #  of many weights, where some are positive and some negative, and the result
            #  is edge_weight * edge_weight_multiplier
            self.edge_weight_multiplier = Parameter(
                torch.Tensor(num_synapses).to(device)
            )
            nn.init.uniform_(self.edge_weight_multiplier, a=-1, b=1)
        if config.train_neurons:
            self.neuron_activation_threshold = Parameter(
                torch.Tensor(num_nodes).to(device)
            )
            nn.init.uniform_(self.neuron_activation_threshold, a=0, b=0.1)

        self.neuron_dropout = nn.Dropout(config.neuron_dropout)

        self.edge_activation_func = self.get_edge_activation_func(
            config.synaptic_limit, config.refined_synaptic_data
        )

    def forward(self, x, edge_index):
        # Start propagating messages.
        size = (x.size(0), x.size(0))

        for _ in range(self.num_passes):
            x = self.propagate(edge_index, size=size, x=x, edge_weight=self.edge_weight)

        return x

    def message(self, x_j, edge_weight):
        # Message: edge_weight (learnable) multiplied by node feature of the neighbor
        # manual reshape to make sure that the multiplication is done correctly
        x_j = x_j.view(self.batch_size, -1)
        if self.train_edges:
            x_j = (
                x_j
                * edge_weight
                * self.edge_activation_func(self.edge_weight_multiplier)
            )
        else:
            x_j = x_j * edge_weight

        # Apply the neuron dropout
        x_j = self.neuron_dropout(x_j)

        return x_j.view(-1, 1)

    def update(self, aggr_out):

        if self.train_neurons:
            # Each node gets its updated feature as the sum of its neighbor contributions.
            # Then, we apply the lambda function with a threshold, to emulate the biological
            temp = aggr_out.view(self.batch_size, -1)
            if self.neuron_normalization == "min_max":
                temp = min_max_norm(temp)
            elif self.neuron_normalization == "log1p":
                temp = log_norm(temp)
            # Apply the threshold. Note the "abs" to make sure that the threshold is not
            #  helping the neuron to activate
            sig_out = self.lambda_func(temp - abs(self.neuron_activation_threshold))
            return sig_out.view(-1, 1)

        return aggr_out

    @staticmethod
    def get_edge_activation_func(synaptic_limit, refined_synaptic_data):
        if synaptic_limit:
            # Here we are multiplying the edge weight by the edge_weight_multiplier. For biological
            # plausibility, we might want to make sure that the edge_weight_multiplier is not bigger than 1
            # Now, if we have the raw synaptic data, all weights are positive, so we need to allow for
            # negative weights, so edge_weight_multiplier will be \in [-1, 1]
            # If we have refined synaptic data, which can have negative weights, the edge_weight_multiplier
            # will be \in [0, 1]
            if refined_synaptic_data:
                return torch.sigmoid
            return torch.tanh
        return nn.Identity()


class FullGraphModel(nn.Module):

    def __init__(self, data_processor, config_, random_generator=None):
        super(FullGraphModel, self).__init__()

        self.connectome = Connectome(data_processor, config_)
        graph_builder = data_processor.graph_builder
        self.train_neurons = config_.train_neurons
        self.register_buffer(
            "decision_making_vector", graph_builder.decision_making_vector
        )
        self.num_features = 1  # only works with 1 for now
        self.batch_size = config_.batch_size
        self.final_layer = config_.final_layer
        self.num_decision_making_neurons = config_.num_decision_making_neurons
        final_layer_input_size = self.compute_final_layer_size(graph_builder, config_)
        self.final_fc = nn.Linear(
            final_layer_input_size, data_processor.num_classes, dtype=config_.dtype
        )
        self.decision_making_dropout = nn.Dropout(config_.decision_dropout)
        self.decision_making_indices = self.select_decision_making_indices(
            graph_builder, config_, random_generator
        )

    def forward(self, data):
        x, edge_index = (
            data.x,
            data.edge_index.long(),
        )  # ensure int64 for PyG operations

        # pass through the connectome and reshape to wide again (batch_size, num_neurons, num_features)
        x = self.connectome(x, edge_index)
        x = x.view(self.batch_size, -1, self.num_features)
        # get final decision
        x = self.decision_making_mask(x)
        # Save the intermediate output for analysis
        if not self.training:
            self.intermediate_output = x.view(self.batch_size, -1).clone().detach()

        if self.final_layer == "mean":
            # get the mean for each batch
            x = torch.mean(x, dim=1, keepdim=True)

        if not self.train_neurons:
            # When we are training edges or only the final layer, the output
            #  explodes a bit and we need to normalize it
            x = x / x.norm()

        x = self.decision_making_dropout(x)

        # final layer to get the correct magnitude
        # Squeeze the num_features. If at some point it is not 1, then we have to change this
        return self.final_fc(x.squeeze(2)).squeeze()

    def decision_making_mask(self, x):
        x = x[:, self.decision_making_vector == 1, :]

        # If we are using a subset of neurons to compute the final decision
        if self.num_decision_making_neurons is not None:
            x = x[:, self.decision_making_indices, :]

        return x

    @staticmethod
    def min_max_norm(x):
        return (x - x.min()) / (x.max() - x.min())

    @staticmethod
    def compute_final_layer_size(graph_builder, config):
        if config.final_layer == "nn":
            if config.num_decision_making_neurons is not None:
                return config.num_decision_making_neurons
            return int(graph_builder.decision_making_vector.sum())
        return 1

    @staticmethod
    def select_decision_making_indices(graph_builder, config_, random_generator):
        if config_.num_decision_making_neurons is not None:
            decision_making_indices = torch.randperm(
                graph_builder.decision_making_vector.sum(),
                generator=random_generator,
                device=config_.DEVICE,
            )[: config_.num_decision_making_neurons]
            return decision_making_indices
        return None
