import random
from sklearn.datasets import make_moons

import torch
import torch.nn as nn
import torch.optim as optim


def get_data(nm_elements, batch_size):
    X, y = make_moons(n_samples=batch_size * (nm_elements // batch_size), noise=0.2, random_state=42)

    train_dl = list(zip(X.reshape(-1, batch_size, 2), y.reshape(-1, batch_size)))

    X_test, y_test = make_moons(n_samples=batch_size * (nm_elements // batch_size) // 2, noise=0.2, random_state=0)
    test_dl = tuple(zip(X_test.reshape(-1, batch_size, 2), y_test.reshape(-1, batch_size)))

    return train_dl, test_dl


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nm_layers, act_fn):
        super(Model, self).__init__()

        self.act_fn = act_fn

        layers = [nn.Linear(input_dim, hidden_dim)]
        layers += [nn.Linear(hidden_dim, hidden_dim) for _ in range(nm_layers - 2)]
        layers += [nn.Linear(hidden_dim, output_dim)]

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act_fn(layer(x))

        x = self.layers[-1](x)

        return x


ce_loss = torch.nn.CrossEntropyLoss()


def train_on_batch(x, y, model, optimizer):
    model.train()
    optimizer.zero_grad()

    y_ = model(x)
    loss = ce_loss(y_, y).sum() / len(x)
    loss.backward()
    optimizer.step()


def eval_on_batch(x, y, model):
    model.eval()

    with torch.no_grad():
        y_ = model(x).argmax(dim=-1)

    return (y_ == y).float().mean(), y_


def train(dl, model, optimizer):
    for x, y in dl:
        train_on_batch(
            torch.from_numpy(x).type(torch.float32), torch.from_numpy(y).type(torch.long), model, optimizer
        )


def eval(dl, model):
    acc = []
    ys_ = []

    for x, y in dl:
        a, y_ = eval_on_batch(torch.from_numpy(x).type(torch.float32), torch.from_numpy(y).type(torch.long), model)
        acc.append(a)
        ys_.append(y_)

    return torch.mean(torch.tensor(acc)), torch.cat(ys_)


batch_size = 32
nm_elements = 1024
nm_epochs = 256 // (nm_elements // batch_size)

model = Model(input_dim=2, hidden_dim=32, output_dim=2, nm_layers=3, act_fn=nn.LeakyReLU())
train_dl, test_dl = get_data(nm_elements, batch_size)

optimizer = optim.AdamW(model.parameters(), lr=1e-2)

for e in range(nm_epochs):
    random.shuffle(train_dl)
    train(train_dl, model=model, optimizer=optimizer)
    a, y = eval(test_dl, model=model)

    print(f"Epoch {e + 1}/{nm_epochs} - Test Accuracy: {a * 100:.2f}%")
