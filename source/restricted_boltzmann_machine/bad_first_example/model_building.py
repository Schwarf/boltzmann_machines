import torch
from data_preprocessing import DataPreprocessing


class RestrictedBoltzmannMachine:
    def __init__(self, number_of_visible_nodes, number_of_hidden_nodes):
        self._weights_hidden_visible = torch.randn(
            number_of_hidden_nodes, number_of_visible_nodes
        )
        self._bias_hidden_nodes = torch.randn(1, number_of_hidden_nodes)
        self._bias_visible_nodes = torch.randn(1, number_of_visible_nodes)

    def hidden_node_sampling_function(self, visible_node_values):
        weighted_values = torch.mm(
            visible_node_values, self._weights_hidden_visible.t()
        )
        activated_values = weighted_values + self._bias_hidden_nodes.expand_as(
            weighted_values
        )
        probability_given_activated_values = torch.sigmoid(activated_values)
        return probability_given_activated_values, torch.bernoulli(
            probability_given_activated_values
        )

    def visible_node_sampling_function(self, hidden_node_values):
        weighted_values = torch.mm(hidden_node_values, self._weights_hidden_visible)
        activated_values = weighted_values + self._bias_visible_nodes.expand_as(
            weighted_values
        )
        probability_given_activated_values = torch.sigmoid(activated_values)
        return probability_given_activated_values, torch.bernoulli(
            probability_given_activated_values
        )

    def train(
        self,
        input_vector,
        visible_nodes_at_kth_iteration,
        vector_of_hidden_nodes_for_input,
        vector_of_hidden_nodes_at_kth_iteration,
    ):
        self._weights_hidden_visible += (
            torch.mm(input_vector.t(), vector_of_hidden_nodes_for_input)
            - torch.mm(
                visible_nodes_at_kth_iteration.t(),
                vector_of_hidden_nodes_at_kth_iteration,
            )
        ).t()
        self._bias_visible_nodes += torch.sum(
            (input_vector - visible_nodes_at_kth_iteration), 0
        )
        self._bias_hidden_nodes += torch.sum(
            (
                vector_of_hidden_nodes_for_input
                - vector_of_hidden_nodes_at_kth_iteration
            ),
            0,
        )


training_data_path = "/media/linux_data/data/movie_data/ml-100k/u1.base"
test_data_path = "/media/linux_data/data/movie_data/ml-100k/u1.test"

data_prep = DataPreprocessing(training_data_path, test_data_path)
training_set, test_set = data_prep.get_data()

visible_nodes = len(training_set[0])
hidden_nodes = 100
batch_size = 100
restricted_boltzmann_machine = RestrictedBoltzmannMachine(visible_nodes, hidden_nodes)

number_of_epochs = 10
for epoch in range(1, number_of_epochs + 1):
    training_loss = 0
    step = 0
    for id_user in range(0, data_prep.number_of_users - batch_size, 100):
        vk = training_set[id_user : id_user + batch_size]
        v0 = training_set[id_user : id_user + batch_size]
        ph0, _ = restricted_boltzmann_machine.hidden_node_sampling_function(v0)
        for k in range(10):
            _, hk = restricted_boltzmann_machine.hidden_node_sampling_function(vk)
            _, vk = restricted_boltzmann_machine.visible_node_sampling_function(hk)
            vk[v0 < 0] = v0[v0 < 0]
        phk, _ = restricted_boltzmann_machine.hidden_node_sampling_function(vk)
    restricted_boltzmann_machine.train(v0, vk, ph0, phk)
    training_loss += torch.mean(torch.abs(v0[v0 > 0] - vk[v0 > 0]))
    step += 1
    print("epoch: " + str(epoch) + " loss: " + str(training_loss / step))
