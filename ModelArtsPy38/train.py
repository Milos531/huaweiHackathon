import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import argparse
import shutil
import torch.onnx
# import onnx

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, 20),
            nn.Softplus(),
            nn.Linear(20, 5),
            nn.Softplus(),
            nn.Linear(5, 1)
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        # print(pred)
        # print(y)
        # exit()
        # pred = torch.round(pred)
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
    correct_true = 0
    correct_false = 0
    incorrect_true = 0
    incorrect_false = 0
    true_cnt = 0
    false_cnt = 0
    with torch.no_grad():
        i = 0
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # print(pred)
            # print(y)

            tens = torch.ones(pred.shape, dtype=torch.float) * 10
            zeros = torch.zeros(pred.shape, dtype=torch.float)
            pred = torch.where(pred > 0.5, tens, zeros)
            # if(10. in y and 10. in pred):
            #     print(y)
            #     print(pred)
            correct += (pred == y).type(torch.float).sum().item()

            for i in range(0, len(pred)):
                if(pred[i].item() == 10. and y[i].item() == 10.):
                    correct_true += 1
                elif(pred[i].item() == 0. and y[i].item() == 0.):
                    correct_false += 1
                elif(pred[i].item() == 10. and y[i].item() == 0.):
                    incorrect_true += 1
                else:
                    incorrect_false += 1

                if(y[i].item() == 10.):
                    true_cnt += 1
                if(y[i].item() == 0.):
                    false_cnt += 1

            # print(correct)
            # exit()
        test_loss /= num_batches
        correct /= size
        print(f"Test error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
        print(f"True accuracy: {100 * correct_true / true_cnt}")
        print(f"False accuracy: {100 * correct_false / false_cnt}")
        print(f"Incorrect true: {100 * incorrect_true / false_cnt}")
        print(f"Incorrect false: {100 * incorrect_false / true_cnt}")
class HeartDataset(Dataset):
    def __init__(self, data, transform = None, target_transorm = None):
        self.data = torch.tensor(data.values[:, :-1], dtype=torch.float)
        self.output_data = torch.tensor(data.values[:, -1:] * 10, dtype=torch.float)
        self.transform = transform
        self.target_transform = target_transorm


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = self.data[idx, 1:]
        output = self.output_data[idx]
        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            output = self.target_transform(output)
        return input, output

def init_data(file):
    data = pd.read_csv(file)
    pd.set_option('display.max_columns', 15)
    # print(data.info())

    data.bmi = data.bmi.fillna(data.bmi.mean())

    # print(data.describe())
    # print(data.describe(include= [object]))
    data.loc[data.gender != "Male", "gender"] = 1
    data.loc[data.gender == "Male", "gender"] = 0
    data.gender = pd.to_numeric(data.gender)

    data.loc[data.ever_married == "Yes", "ever_married"] = 1
    data.loc[data.ever_married == "No", "ever_married"] = 0
    data.ever_married = pd.to_numeric(data.ever_married)

    data.loc[data.work_type == "Private", "work_type"] = 0
    data.loc[data.work_type == "Self-employed", "work_type"] = 1
    data.loc[data.work_type == "Never_worked", "work_type"] = 2
    data.loc[data.work_type == "children", "work_type"] = 3
    data.loc[data.work_type == "Govt_job", "work_type"] = 4
    data.work_type = pd.to_numeric(data.work_type)

    data.loc[data.Residence_type == "Urban", "Residence_type"] = 0
    data.loc[data.Residence_type == "Rural", "Residence_type"] = 1
    data.Residence_type = pd.to_numeric(data.Residence_type)

    data.loc[data.smoking_status == "never smoked", "smoking_status"] = 0
    data.loc[data.smoking_status == "Unknown", "smoking_status"] = 1
    data.loc[data.smoking_status == "formerly smoked", "smoking_status"] = 1
    data.loc[data.smoking_status == "smokes", "smoking_status"] = 2
    data.smoking_status = pd.to_numeric(data.smoking_status)

    # print(data.head())
    # print(data.info())
    # print(data.describe(include= [object]))
    # print(data.describe())

    # print(len(data))

    data = data.sample(frac=1, random_state=1)
    return data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train-url', type= str, help='the path model saved')
    parser.add_argument('--data-url', type= str, help= 'the training data')
    args, unknown = parser.parse_known_args()

    data = init_data(args.data_url)

    training_data = HeartDataset(
        data[:int(0.8 * (len(data)))],)
    test_data = HeartDataset(data[int(0.8*len(data)):])

    #print(training_data.data)
    #print(test_data.data)

    train_dataloader = DataLoader(training_data, batch_size=10)
    test_dataloader = DataLoader(test_data, batch_size=10)

    model = NeuralNetwork()
    learning_rate = 1e-3
    batch_size = 10
    epochs = 10

    loss_fn = nn.MSELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

    for i in range(epochs):
        print(f"Epoch {i + 1}:\n-------------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)


    print("Done!")

    model_path = os.path.join(args.train_url, "model")
    os.makedirs(model_path, exist_ok=True)

    # model_scripted = torch.jit.script(model)
    # model_scripted.save(os.path.join(model_path,"model_weights.pt"))
    torch.save(model.state_dict(), os.path.join(model_path,"model_weights.pt"))

    # torch.onnx.export(model, torch.ones(10, dtype=torch.float), os.path.join(model_path, "model_weights.onnx"))

    current_file_path = os.path.dirname(__file__)
    shutil.copyfile(os.path.join(current_file_path, 'infer/customize_service.py'), os.path.join(model_path, 'customize_service.py'))
    shutil.copyfile(os.path.join(current_file_path, 'infer/config.json'), os.path.join(model_path, 'config.json'))
