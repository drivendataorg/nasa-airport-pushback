import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    # Implement a multi linear perceptron with single hidden layer
    # This is a base regressor
    def __init__(self, d, h):
        super(MLP, self).__init__()
        # hidden layer
        self.linear_one = nn.Linear(d, h)
        self.linear_two = nn.Linear(h, 1)
        self.act_func = nn.ReLU()
        self.layer_out = nn.ReLU()

    # prediction function
    def forward(self, x):
        out = self.linear_one(x)
        out = self.linear_two(self.act_func(out))
        # Since the output is a non-negative number (minutes until pushback)
        # so we use ReLu activation function to make sure output is non-negative
        return self.layer_out(out)


def train_eval(net, trainloader, x_test, y_test, options):
    """
    Train a neural network regressor
    trainloader: Torch training data loader
    x_test : input features of test set=> torch Tensor data type.
    y_test: label of testset => torch Tensor data type
    """
    lr = options.get("lr", 1e-3)
    epochs = options.get("epochs", 200)
    criterion = nn.L1Loss()
    error_list = []
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    for epoch in range(epochs):
        for x_train, y_train in trainloader:
            net.train()
            optimizer.zero_grad()
            y_pred = net(x_train)
            loss = criterion(y_pred, y_train.reshape(-1, 1))
            loss.backward()
            optimizer.step()
        if options["eval"]:
            net.eval()
            y_pred = net(x_test)
            loss = criterion(y_pred, y_test.reshape(-1, 1))
            error_list.append(loss.item())
            if epoch % 20 == 0:
                print(epoch, loss.item())

    return net, error_list
