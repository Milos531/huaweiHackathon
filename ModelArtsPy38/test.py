import torch
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from train import test_loop, HeartDataset, init_data, NeuralNetwork




if __name__ == "__main__":
    data = init_data("data/healthcare-dataset-stroke-data.csv")

    dataset = HeartDataset(data)
    dataloader = DataLoader(dataset, batch_size=10)
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model_weights.pth"))
    test_loop(dataloader, model, nn.MSELoss())