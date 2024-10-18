import warnings
import xgboost as xgb
from flwr.client import ClientApp, Client, start_client
from flwr.common import Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes, Status, Code
from dotenv import load_dotenv
from utils.classification.xgboost_classification import load_data, replace_keys, unflatten_dict
import os

warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv('../.env')

CLIENT_TRAINING_SET = os.getenv('CLIENT_3_TRAINING_SET')
CLIENT_TESTING_SET = os.getenv('CLIENT_3_TESTING_SET')
local_epochs = 100  

class FlowerClient(Client):
    def __init__(self, train_dmatrix, valid_dmatrix, num_train, num_val, num_local_round, params):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params

    def _local_boost(self, bst_input):
        for i in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())
        
        bst = bst_input[
            bst_input.num_boosted_rounds() - self.num_local_round : bst_input.num_boosted_rounds()
        ]
        return bst

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        
        if global_round == 1:
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")]
            )
        else:
            bst = xgb.Booster(params=self.params)
            global_model = bytearray(ins.parameters.tensors[0])
            bst.load_model(global_model)
            bst = self._local_boost(bst)
        
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)
        
        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        bst = xgb.Booster(params=self.params)
        para_b = bytearray(ins.parameters.tensors[0])
        bst.load_model(para_b)

        eval_results = bst.eval_set(evals=[(self.valid_dmatrix, "valid")], iteration=bst.num_boosted_rounds() - 1)
        # auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
        eval_list = eval_results.split("\t")
        print(eval_list)
        auc = round(float(eval_list[1].split(':')[1]),4)
        loss = round(float(eval_list[2].split(':')[1]),4)

        print(f"Client validation AUC: {auc}")
        print(f"Client validation Loss: {loss}")

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=loss,
            num_examples=self.num_val,
            metrics={"AUC": auc, "LOSS":loss},
        )

def client_fn(context):
    partition_id = 3
    num_partitions = 3
    train_dmatrix, valid_dmatrix, num_train, num_val = load_data(partition_id, num_partitions)

    # cfg = replace_keys(unflatten_dict(context.run_config))
    num_local_round = 50

    return FlowerClient(
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        params={
            'objective': 'binary:logistic',
            'eta': 0.05,  # Learning rate
            'max_depth': 6,
            'eval_metric': ['auc', 'logloss'],
            'nthread': 16,
            'num_parallel_tree': 1,
            'subsample': 1,
            'tree_method': 'hist'
        }
    )

# app = ClientApp(client_fn)

start_client(
    server_address='127.0.0.1:8000',
    client_fn=client_fn
)
