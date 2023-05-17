import os
import sys

sys.path.append(os.getcwd())
from src import *


from client import FL_Client, CNN, test, tree_encoding_loader
from server import FL_Server
from make_dataset import do_fl_partitioning, make_public_dataset

airport = "KATL"

""" 
Below code modified from tutorial code provided by paper:
Ma, C., Qiu, X., Beutel, D., and Lane, N. Gradient-less
federated gradient boosting tree with learnable learn-
ing rates. In Proceedings of the 3rd Workshop on Ma-
chine Learning and Systems. ACM, may 2023. doi: 10.
1145/3578356.3592579. 
"""


class SaveModelStrategy(fl.server.strategy.FedXgbNnAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):

        # Call aggregate_fit from base class (FedXgbNnAvg) to aggregate parameters and metrics
        aggregated_model, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        parameters_aggregated, trees_aggregated = aggregated_model

        if aggregated_model is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarray_weights: List[
                np.ndarray
            ] = fl.common.parameters_to_ndarrays(parameters_aggregated)
            if server_round == 1:
                os.makedirs(f"trees/{airport}", exist_ok=True)
                for tree in trees_aggregated:
                    tree[0].save_model(f"trees/{airport}/tree-{tree[1]}")

            print(f"Saving round {server_round} aggregated_ndarrays...")
            os.makedirs(f"model/{airport}", exist_ok=True)
            np.save(
                f"model/{airport}/round-{server_round}-weights.npy",
                aggregated_ndarray_weights,
            )

        return aggregated_model, aggregated_metrics


def serverside_eval(
    server_round: int,
    parameters: Tuple[
        Parameters,
        Union[Tuple[CatBoostRegressor, int], List[Tuple[CatBoostRegressor, int]],],
    ],
    config: Dict[str, Scalar],
    testloader: DataLoader,
    batch_size: int,
    client_tree_num: int,
    client_num: int,
) -> Tuple[float, Dict[str, float]]:
    """An evaluation function for centralized/serverside evaluation over the entire test set."""
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = CNN(client_num, client_tree_num)
    # print_model_layers(model)

    model.set_weights(parameters_to_ndarrays(parameters[0]))
    model.to(device)

    trees_aggregated = parameters[1]
    testloader = tree_encoding_loader(
        testloader, batch_size, trees_aggregated, client_tree_num, client_num
    )
    loss, result, _ = test(model, testloader, device=device, log_progress=False)
    print(f"Evaluation on the server: test_loss={loss:.4f}, test_mse={result:.4f}")
    return loss, {"mae": result}


def start_experiment(
    a_port,
    dataset,
    num_rounds: int = 5,
    client_tree_num: int = 50,
    num_iterations: int = 100,
    fraction_fit: float = 1.0,
    batch_size: int = 32,
    val_ratio: float = 0.0,
) -> History:
    client_resources = {"num_cpus": 0.5}  # 2 clients per CPU

    global airport
    airport = a_port

    # Partition the dataset into subsets reserved for each client.
    # - 'val_ratio' controls the proportion of the (local) client reserved as a local test set
    # (good for testing how the final model performs on the client's local unseen data)
    # def do_fl_partitioning(dataset, batch_size: Union[int, str], val_ratio: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    trainloaders, valloaders, testloader = do_fl_partitioning(
        dataset, airport, batch_size="whole", test_ratio=0.05,
    )

    client_pool_size = len(trainloaders)
    min_fit_clients = client_pool_size // 4

    # Configure the strategy
    def fit_config(server_round: int) -> Dict[str, Scalar]:
        print(f"Configuring round {server_round}")
        return {
            "num_iterations": num_iterations,
            "batch_size": batch_size,
        }

    # FedXgbNnAvg
    strategy = SaveModelStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit if val_ratio > 0.0 else 0.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_fit_clients,
        min_available_clients=client_pool_size,  # all clients should be available
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=(lambda r: {"batch_size": batch_size}),
        evaluate_fn=functools.partial(
            serverside_eval,
            testloader=testloader,
            batch_size=batch_size,
            client_tree_num=client_tree_num,
            client_num=client_pool_size,
        ),
        accept_failures=False,
    )

    print(
        f"FL round will proceed with {fraction_fit * 100}% of clients sampled, at least {min_fit_clients}."
    )

    def client_fn(cid: str) -> fl.client.Client:
        """Creates a federated learning client"""
        if val_ratio > 0.0 and val_ratio <= 1.0:
            return FL_Client(
                trainloaders[int(cid)],
                valloaders[int(cid)],
                client_tree_num,
                client_pool_size,
                cid,
                log_progress=False,
            )
        else:
            return FL_Client(
                trainloaders[int(cid)],
                None,
                client_tree_num,
                client_pool_size,
                cid,
                log_progress=False,
            )

    # Start the simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        server=FL_Server(client_manager=SimpleClientManager(), strategy=strategy),
        num_clients=client_pool_size,
        client_resources=client_resources,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    print(history)

    return history


if __name__ == "__main__":
    for airport in airports:
        dataset = make_public_dataset(airport, True)
        print(
            start_experiment(
                a_port=airport,
                dataset=dataset,
                num_rounds=5,
                client_tree_num=client_tree_num,
                num_iterations=50,
                batch_size=64,
                fraction_fit=1.0,
                val_ratio=0.0,
            )
        )
