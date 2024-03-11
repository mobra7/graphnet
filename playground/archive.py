import torch
import torch.nn as nn
import torch.optim as optim
from graphnet.models.task import StandardLearnedTask
from graphnet.training.loss_functions import BinaryCrossEntropyLoss


class approximate_likelihood(StandardLearnedTask):

    # classes: shuffled and non shuffled
    default_target_labels = ["class"]  
    default_prediction_labels = ["likelihood_pred"]
    nb_inputs = 10

    def __init__(self, hidden_size, hidden_layer_size, output_size, learning_rate=0.01):
        super().__init__(hidden_size=hidden_size,loss_function=BinaryCrossEntropyLoss)
        self.fc1 = nn.Linear(hidden_size, hidden_layer_size) # hidden_size will be nb_inputs eventually?
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.BCE = BinaryCrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

    def _forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    # which parameters does the optimizer need exactly 
    # only fc or also info about which nonlinearities are used?
    def parameters(self):
        return list(self.fc1.parameters()) + list(self.fc2.parameters())

    def train(self, inputs, targets, epochs=100):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.BCE(outputs, targets)
            loss.backward()
            self.optimizer.step()
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

    def create_training_data(self, unshuffled_data, random_seed = 0):
        torch.manual_seed(random_seed)
        n, k = unshuffled_data.shape
        shuffled_feature_idx = k-1
        
        shuffled_data = unshuffled_data.clone()
        shuffled_data[:, shuffled_feature_idx] = shuffled_data[torch.randperm(n), shuffled_feature_idx]

        training_features = torch.cat((unshuffled_data, shuffled_data), dim=0)
        training_labels = torch.cat((torch.ones(n), torch.zeros(n)))
        return training_features, training_labels.unsqueeze(1)
    

if __name__ == "__main__":
    input_size = 10  
    hidden_size = 20  
    output_size = 1
    model = approximate_likelihood(input_size, hidden_size, output_size)

    unshuffled_data = torch.randn(1000, input_size)

    X_train, y_train = model.create_training_data(unshuffled_data)

    model.train(X_train, y_train)

# todo: 
# how can I specify nb_inputs within __init__
# make create_training_data work with some actual data
# make it work with two features to be shuffled
# concat features and labels for training?
# train both model and task independently?
# whats a good ratio of shuffled and non shuffled?
# will i need output size?
# try implementing with actual model
# approximate likelihood for different angles within task or "from outside"
# make number of inputs flexible
# make model and training more advanced and flexible

###################################################################################

