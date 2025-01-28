import os
import torch
from torch import nn

from model_service.pytorch_model_service import PTServingBaseService

class PTServingTextService(PTServingBaseService):

    def __init__(self, model_name, model_path):
        super(PTServingTextService, self).__init__(model_name, model_path)


        self.model = Mnist(model_path)


    def _preprocess(self, data):
        x = torch.ones(10, dtype=torch.float)
        x[0] = 0 if data["gender"] == "Male" else 1
        x[1] = data["age"]
        x[2] = data["hypertension"]
        x[3] = data["heart_disease"]
        x[4] = 0 if data["ever_married"] == "No" else 1
        x[5] = 0 if data["work_type"] == "Private" else \
            1 if data["work_type"] == "Self-emplyed" else \
                2 if data["work_type"] == "Never_worked" else \
                    3 if data["work_type"] == "children" else 4
        x[6] = 0 if data["Residence_type"] else 1
        x[7] = data["avg_glucose_level"]
        x[8] = data["bmi"]
        x[9] = 0 if data["smoking_status"] == "never smoked" else \
            1 if data["smoking_status"] == "Unknown" or data["smoking_status"] == "formerly smoked" else 2
        return x

    def _inference(self, data):
        return self.model(data)

    def _postprocess(self, data):
        result = 0 if data < 0.5 else 1
        return result

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

def Mnist(model_path, **kwargs):
    model = NeuralNetwork()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        # model = torch.jit.load(model_path)
        model.load_state_dict(torch.load(model_path, map_location="cuda:0"))
    else:
        device = torch.device('cpu')
        # model = torch.jit.load(model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))


    model.to(device)

    model.eval()

    return model