from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvg

from util.utils2 import *
import torch

# load the testset from ../testing
testset = torch.load('../testing/testset.pt')
testset_137 = torch.load('../testing/model1_testset.pt')
testset_258 = torch.load('../testing/model2_testset.pt')
testset_469 = torch.load('../testing/model3_testset.pt')


"""
This function allows you to set (or load) the parameters (weights and biases) of the model manually. 
This can be useful if you want to initialize the model with specific weights, such as pre-trained weights.
"""

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in params_dict}
    )
    net.load_state_dict(state_dict, strict=True)

"""
This function allows you to retrieve (or save) the current parameters (weights and biases) of the model. 
This can be useful for inspecting what the model has learned or for saving the model state for later use
"""
def get_weights(net):
    ndarrays = [
        val.cpu().numpy() for _, val in net.state_dict().items()
    ]
    return ndarrays

"""
facilitate automatic model transfer at the end of training.
or
Use a shared storage system accessible by both the server and clients to save and retrieve the model.
"""
def transfer_final_model_to_clients(model_path):
    pass

def evaluate(server_round, parameters, config):
    net = SimpleModel()
    set_weights(net, parameters)

    _, accuracy = evaluate_model(net, testset)
    _, accuracy137 = evaluate_model(net, testset_137)
    _, accuracy258 = evaluate_model(net, testset_258)
    _, accuracy469 = evaluate_model(net, testset_469)

    log(INFO, "test accuracy on all digits: %.4f", accuracy)
    log(INFO, "test accuracy on [1,3,7]: %.4f", accuracy137)
    log(INFO, "test accuracy on [2,5,8]: %.4f", accuracy258)
    log(INFO, "test accuracy on [4,6,9]: %.4f", accuracy469)

    if server_round == 3:
        model_save_path = f"../models/global/final_round_global_model.pth"
        torch.save(net.state_dict(), model_save_path)
        cm = compute_confusion_matrix(net, testset)
        plot_confusion_matrix(cm, "Final Global Model", "../plots/confusion_matrix/final_global_model.png")
        transfer_final_model_to_clients(model_save_path)
    else:
        model_save_path = f"../models/global/{server_round}_round_global_model.pth"
        torch.save(net.state_dict(), model_save_path)
        confusion_matrix_name = str(server_round) + "_round_global_model.png"
        cm = compute_confusion_matrix(net, testset)
        plot_confusion_matrix(cm, "Global Model", "../plots/confusion_matrix/"+str(confusion_matrix_name))


net = SimpleModel()
params = ndarrays_to_parameters(get_weights(net))


# Start Flower server
start_server(
  server_address="0.0.0.0:8000",
  config=ServerConfig(num_rounds=3),
  strategy=FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        initial_parameters=params,
        evaluate_fn=evaluate,
        min_fit_clients=3,      
        min_evaluate_clients=3,  
        min_available_clients=3,
    )
)