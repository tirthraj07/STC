from flwr.client import NumPyClient, start_numpy_client

from util.utils2 import *
import torch

# load the training dataset from ../training
trainset = torch.load('../training/trainset.pt')
part1 = torch.load('../training/model1_trainset.pt')
part2 = torch.load('../training/model2_trainset.pt')
part3 = torch.load('../training/model3_trainset.pt')

train_sets = [part1, part2, part3]

testset = torch.load('../testing/testset.pt')

# Sets the parameters of the model
def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in params_dict}
    )
    net.load_state_dict(state_dict, strict=True)

# Retrieves the parameters from the model
def get_weights(net):
    ndarrays = [
        val.cpu().numpy() for _, val in net.state_dict().items()
    ]
    return ndarrays

class FlowerClient(NumPyClient):
    def __init__(self, net, trainset, testset):
        self.net = net
        self.trainset = trainset
        self.testset = testset
        self.round_number = 0

    # Train the model
    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_model(self.net, self.trainset)
        model_save_path = f"../models/client/client2/round_{self.round_number}_model.pth"
        torch.save(self.net.state_dict(), model_save_path)
        self.round_number += 1
        return get_weights(self.net), len(self.trainset), {}

    # Test the model
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        set_weights(self.net, parameters)
        loss, accuracy = evaluate_model(self.net, self.testset)
        return loss, len(self.testset), {"accuracy": accuracy}
    

net = SimpleModel()
partition_id =  2;
client_train = train_sets[int(partition_id)]
client_test = testset
start_numpy_client(server_address="127.0.0.1:8000", client=FlowerClient(net, client_train, client_test))