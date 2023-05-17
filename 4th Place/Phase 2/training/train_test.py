import torch
from config import *


def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for X, y in trainloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(X)
            total += 1
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss

        epoch_loss /= total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.L1Loss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for X, y in testloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = net(X)
            loss += criterion(outputs, y).item()
            total += 1

    loss /= total
    return loss
