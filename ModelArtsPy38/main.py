import os
import torch
from torch import nn

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            print(pred)
            print(y)
            exit()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")


if __name__ == '__main__':
    # training_data = datasets.FashionMNIST(
    #     root="data",
    #     train = True,
    #     download= True,
    #     transform=ToTensor()
    # )
    #
    # test_data = datasets.FashionMNIST(
    #     root="data",
    #     train = False,
    #     download= True,
    #     transform= ToTensor()
    # )
    #
    # labels_map = {
    #     0: "T-Shirt",
    #     1: "Trouser",
    #     2: "Pullover",
    #     3: "Dress",
    #     4: "Coat",
    #     5: "Sandal",
    #     6: "Shirt",
    #     7: "Sneaker",
    #     8: "Bag",
    #     9: "Ankle Boot"
    # }
    #
    # figure = plt.figure(figsize=(8, 8))
    #
    # cols, rows = 3, 3
    #
    # for i in range(1, cols*rows + 1):
    #     sample_idx = torch.randint(len(training_data), size=(1,)).item()
    #     img, label = training_data[sample_idx]
    #     figure.add_subplot(rows, cols, i)
    #     plt.title(labels_map[label])
    #     plt.axis("off")
    #     plt.imshow(img.squeeze(), cmap="gray")
    # plt.show()

    # ds = datasets.FashionMNIST(
    #     root="data",
    #     train = True,
    #     download= True,
    #     transform = ToTensor(),
    #     target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    #
    # )

    #Neural Network (Model making)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # #print(f"Using {device} device")
    #
    # model = NeuralNetwork().to(device)
    # print(model)
    #
    # X = torch.rand(1, 28, 28, device= device)
    # logits = model(X)
    # pred_probab = nn.Softmax(dim=1)(logits)
    # y_pred = pred_probab.argmax(1)
    # print(f"Predicted class: {y_pred}")
    #
    #
    # input_image = torch.rand(3, 28, 28)
    # print(input_image.size())
    #
    # #nn.Flatten
    # flatten = nn.Flatten()
    # flat_image = flatten(input_image)
    # print(flat_image.size())
    #
    # #nn.Linear
    # layer1 = nn.Linear(in_features=28*28, out_features=20)
    # hidden1 = layer1(flat_image)
    # print(hidden1.size())
    #
    # #nn.ReLU
    # print(f"Before ReLU: {hidden1}")
    # hidden1 = nn.ReLU()(hidden1)
    # print(f"After ReLU: {hidden1}")
    #
    # #nn.Sequential
    # seq_modules = nn.Sequential( flatten, layer1, nn.ReLU(), nn.Linear(20, 10))
    # input_image = torch.rand(3, 28, 28)
    # logits = seq_modules(input_image)
    # print(logits)
    #
    #
    #
    # #nnSoftmax
    # softmax = nn.Softmax(dim = 1)
    # pred_probab = softmax(logits)
    # print(pred_probab.argmax(1))
    #
    #
    # #Model parameters
    # print(f"Model structure: {model}\n\n")
    #
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")


    # x = torch.ones(5) #input
    # y = torch.zeros(3) #output
    # w = torch.randn(5, 3, requires_grad=True)
    # b = torch.randn(3, requires_grad=True)
    # z = torch.matmul(x, w) + b
    # loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    # print(loss)
    #
    # loss.backward()
    # print(w.grad)
    # print(b.grad)
    # z= torch.matmul(x, w) + b
    # loss= torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    # print(loss)

    training_data = datasets.FashionMNIST(
        root="data",
        train = True,
        download=True,
        transform=ToTensor()
    )

    test_data =datasets.FashionMNIST(
        root="data",
        train = False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    model = NeuralNetwork()
    learning_rate = 1e-3
    batch_size =64
    epochs = 10

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n--------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)


    print("Done!")

