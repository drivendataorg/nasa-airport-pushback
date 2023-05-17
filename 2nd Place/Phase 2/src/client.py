from src import *


def construct_tree(
    dataset: Dataset, label: NDArray, n_estimators: int
) -> CatBoostRegressor:
    """Construct a catboost tree from tabular dataset."""

    train_pool = Pool(np.array(dataset, copy=True), np.array(label, copy=True))

    hyper_params = {
        "iterations": n_estimators,
        "learning_rate": 0.15,
        "l2_leaf_reg": 15,
        "loss_function": "MAE",
        "thread_count": -1,
        "metric_period": 50,
        "max_depth": 9,
        "max_bin": 63,
        "eval_metric": "MAE",
    }

    tree = CatBoostRegressor(**hyper_params)
    tree.fit(train_pool)

    return tree


""" 
Below code modified from tutorial code provided by paper:
Ma, C., Qiu, X., Beutel, D., and Lane, N. Gradient-less
federated gradient boosting tree with learnable learn-
ing rates. In Proceedings of the 3rd Workshop on Ma-
chine Learning and Systems. ACM, may 2023. doi: 10.
1145/3578356.3592579. 
"""


def construct_tree_from_loader(
    dataset_loader: DataLoader, n_estimators: int
) -> CatBoostRegressor:
    """Construct a catboost tree form tabular dataset loader."""
    for dataset in dataset_loader:
        data, label = dataset[0], dataset[1]
    return construct_tree(data, label, n_estimators)


def tree_encoding(  # pylint: disable=R0914
    trainloader: DataLoader,
    client_trees: Union[CatBoostRegressor, List[Tuple[CatBoostRegressor, int]],],
    client_tree_num: int,
    client_num: int,
) -> Optional[Tuple[NDArray, NDArray]]:
    """Transform the tabular dataset into prediction results using the
    aggregated xgboost tree ensembles from all clients."""
    if trainloader is None:
        return None

    for local_dataset in trainloader:
        x_train, y_train = local_dataset[0], local_dataset[1]

    x_train_enc = np.zeros((x_train.shape[0], client_num * client_tree_num))
    x_train_enc = np.array(x_train_enc, copy=True)

    temp_trees: Any = None
    if isinstance(client_trees, list) is False:
        temp_trees = [client_trees[0]] * client_num
    elif isinstance(client_trees, list) and len(client_trees) != client_num:
        temp_trees = [client_trees[0][0]] * client_num
    else:
        cids = []
        temp_trees = []
        for i, _ in enumerate(client_trees):
            temp_trees.append(client_trees[i][0])  # type: ignore
            cids.append(client_trees[i][1])  # type: ignore
        sorted_index = np.argsort(np.asarray(cids))
        temp_trees = np.asarray(temp_trees)[sorted_index]

    for i, _ in enumerate(temp_trees):
        for j, pred in enumerate(tree_predictions(temp_trees[i], x_train)):
            x_train_enc[:, i * client_tree_num + j] = pred
        # for j in range(client_tree_num):
        #     x_train_enc[:, i * client_tree_num + j] = single_tree_prediction(
        #         temp_trees[i], j, x_train
        #     )

    x_train_enc32: Any = np.float32(x_train_enc)
    y_train32: Any = np.float32(y_train)

    x_train_enc32, y_train32 = (
        torch.from_numpy(
            np.expand_dims(x_train_enc32, axis=1)  # type: ignore
        ),
        torch.from_numpy(
            np.expand_dims(y_train32, axis=-1)  # type: ignore
        ),
    )
    return x_train_enc32, y_train32


def tree_encoding_loader(
    dataloader: DataLoader,
    batch_size: int,
    client_trees: Union[
        Tuple[CatBoostRegressor, int], List[Tuple[CatBoostRegressor, int]]
    ],
    client_tree_num: int,
    client_num: int,
) -> DataLoader:
    encoding = tree_encoding(dataloader, client_trees, client_tree_num, client_num)
    if encoding is None:
        return None
    data, labels = encoding
    tree_dataset = TreeDataset(data, labels)
    return get_dataloader(tree_dataset, "tree", batch_size)


def tree_predictions(tree: CatBoostRegressor, dataset: NDArray) -> Optional[NDArray]:
    """Extract the prediction result of a single tree in the catboost tree
    ensemble."""
    num_t = tree.tree_count_
    # print(dataset.shape)
    return tree.staged_predict(
        np.array(dataset, copy=True), ntree_start=0, ntree_end=num_t
    )


def single_tree_prediction(
    tree: CatBoostRegressor, n_tree: int, dataset: NDArray
) -> Optional[NDArray]:
    """Extract the prediction result of a single tree in the catboost tree
    ensemble."""
    num_t = tree.tree_count_
    if n_tree > num_t - 1:
        print(
            "The tree index to be extracted is larger than the total number of trees."
        )
        return None

    return next(
        tree.staged_predict(
            dataset.detach().cpu().numpy(), ntree_start=n_tree, ntree_end=n_tree + 1
        )
    )


def get_dataloader(
    dataset: Dataset, partition: str, batch_size: Union[int, str]
) -> DataLoader:
    if batch_size == "whole":
        batch_size = len(dataset)
    return DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, shuffle=(partition == "train")
    )


class CNN(nn.Module):
    def __init__(self, client_num, client_tree_num, n_channel: int = 64) -> None:
        super(CNN, self).__init__()
        n_out = 1
        self.conv1d = nn.Conv1d(
            1, n_channel, kernel_size=client_tree_num, stride=client_tree_num, padding=0
        )
        self.layer_direct = nn.Linear(n_channel * client_num, n_out)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        self.Identity = nn.Identity()

        # Add weight initialization
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(
                    layer.weight, mode="fan_in", nonlinearity="relu"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ReLU(self.conv1d(x))
        x = x.flatten(start_dim=1)
        x = self.ReLU(x)
        x = self.Identity(self.layer_direct(x))
        return x

    def get_weights(self) -> fl.common.NDArrays:
        """Get model weights as a list of NumPy ndarrays."""
        return [
            np.array(val.cpu().numpy(), copy=True)
            for _, val in self.state_dict().items()
        ]

    def set_weights(self, weights: fl.common.NDArrays) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        layer_dict = {}
        for k, v in zip(self.state_dict().keys(), weights):
            if v.ndim != 0:
                layer_dict[k] = torch.Tensor(np.array(v, copy=True))
        state_dict = OrderedDict(layer_dict)
        self.load_state_dict(state_dict, strict=True)


def train(
    net: CNN,
    trainloader: DataLoader,
    device: torch.device,
    num_iterations: int,
    log_progress: bool = True,
) -> Tuple[float, float, int]:
    # Define loss and optimizer
    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-6)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999))

    def cycle(iterable):
        """Repeats the contents of the train loader, in case it gets exhausted in 'num_iterations'."""
        while True:
            for x in iterable:
                yield x

    # Train the network
    net.train()
    total_loss, total_result, n_samples = 0.0, 0.0, 0
    pbar = (
        tqdm(iter(cycle(trainloader)), total=num_iterations, desc=f"TRAIN")
        if log_progress
        else iter(cycle(trainloader))
    )

    for i, data in zip(range(num_iterations), pbar):
        tree_outputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = net(tree_outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Collected training loss and accuracy statistics
        total_loss += loss.item()
        n_samples += labels.size(0)

        mse = MeanAbsoluteError()(outputs, labels.type(torch.int))
        total_result += mse * labels.size(0)

        if log_progress:
            pbar.set_postfix(
                {
                    "train_loss": total_loss / n_samples,
                    "train_mse": total_result / n_samples,
                }
            )
    if log_progress:
        print("\n")

    return total_loss / n_samples, total_result / n_samples, n_samples


def test(
    net: CNN, testloader: DataLoader, device: torch.device, log_progress: bool = True,
) -> Tuple[float, float, int]:
    """Evaluates the network on test data."""
    criterion = nn.MSELoss()

    total_loss, total_result, n_samples = 0.0, 0.0, 0
    net.eval()
    with torch.no_grad():
        pbar = tqdm(testloader, desc="TEST") if log_progress else testloader
        for data in pbar:
            tree_outputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(tree_outputs)

            # Collected testing loss and accuracy statistics
            total_loss += criterion(outputs, labels).item()
            n_samples += labels.size(0)

            mae = MeanAbsoluteError()(outputs.cpu(), labels.type(torch.int).cpu())
            total_result += mae * labels.size(0)

    if log_progress:
        print("\n")

    return total_loss / n_samples, total_result / n_samples, n_samples


# Flower client


class FL_Client(fl.client.Client):
    def __init__(
        self,
        trainloader: DataLoader,
        valloader: DataLoader,
        client_tree_num: int,
        client_num: int,
        cid: str,
        log_progress: bool = False,
    ):
        """
        Creates a client for training `network.Net` on tabular dataset.
        """
        self.cid = cid
        self.tree = construct_tree_from_loader(trainloader, client_tree_num)
        self.trainloader_original = trainloader
        self.valloader_original = valloader
        self.trainloader = None
        self.valloader = None
        self.client_tree_num = client_tree_num
        self.client_num = client_num
        self.properties = {"tensor_type": "numpy.ndarray"}
        self.log_progress = log_progress

        # instantiate model
        self.net = CNN(client_num, client_tree_num)

        # determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return GetPropertiesRes(properties=self.properties)

    def get_parameters(
        self, ins: GetParametersIns
    ) -> Tuple[GetParametersRes, Tuple[CatBoostRegressor, int]]:
        return [
            GetParametersRes(
                status=Status(Code.OK, ""),
                parameters=ndarrays_to_parameters(self.net.get_weights()),
            ),
            (self.tree, int(self.cid)),
        ]

    def set_parameters(
        self,
        parameters: Tuple[
            Parameters,
            Union[Tuple[CatBoostRegressor, int], List[Tuple[CatBoostRegressor, int]],],
        ],
    ) -> Union[
        Tuple[CatBoostRegressor, int], List[Tuple[CatBoostRegressor, int]],
    ]:
        self.net.set_weights(parameters_to_ndarrays(parameters[0]))
        return parameters[1]

    def fit(self, fit_params: FitIns) -> FitRes:
        # Process incoming request to train
        num_iterations = fit_params.config["num_iterations"]
        batch_size = fit_params.config["batch_size"]
        aggregated_trees = self.set_parameters(fit_params.parameters)

        if type(aggregated_trees) is list:
            print("Client " + self.cid + ": recieved", len(aggregated_trees), "trees")
        else:
            print("Client " + self.cid + ": only had its own tree")

        # if self.trainloader is None:
        self.trainloader = tree_encoding_loader(
            self.trainloader_original,
            batch_size,
            aggregated_trees,
            self.client_tree_num,
            self.client_num,
        )

        # if self.valloader is None:
        self.valloader = tree_encoding_loader(
            self.valloader_original,
            batch_size,
            aggregated_trees,
            self.client_tree_num,
            self.client_num,
        )

        # num_iterations = None special behaviour: train(...) runs for a single epoch, however many updates it may be
        num_iterations = num_iterations or len(self.trainloader)

        # Train the model
        print(f"Client {self.cid}: training for {num_iterations} iterations/updates")
        self.net.to(self.device)
        train_loss, train_result, num_examples = train(
            self.net,
            self.trainloader,
            device=self.device,
            num_iterations=num_iterations,
            log_progress=self.log_progress,
        )
        print(
            f"Client {self.cid}: training round complete, {num_examples} examples processed"
        )

        # Return training information: model, number of examples processed and metrics
        return FitRes(
            status=Status(Code.OK, ""),
            parameters=self.get_parameters(fit_params.config),
            num_examples=num_examples,
            metrics={"loss": train_loss, "mse": train_result},
        )

    def evaluate(self, eval_params: EvaluateIns) -> EvaluateRes:
        # Process incoming request to evaluate
        self.set_parameters(eval_params.parameters)

        # Evaluate the model
        self.net.to(self.device)
        loss, result, num_examples = test(
            self.net,
            self.valloader,
            device=self.device,
            log_progress=self.log_progress,
        )

        # Return evaluation information
        print(
            f"Client {self.cid}: evaluation on {num_examples} examples: loss={loss:.4f}, mse={result:.4f}"
        )
        return EvaluateRes(
            status=Status(Code.OK, ""),
            loss=loss,
            num_examples=num_examples,
            metrics={"mse": result},
        )
