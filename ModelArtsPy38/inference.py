from torch import nn
import torch
# import onnx
# from onnx2pytorch import ConvertModel

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

model = NeuralNetwork()
model.load_state_dict(torch.load("model/model_weights.pt"))
model.eval()

def inference(data):
    x = torch.ones(10, dtype=torch.float)
    x[0] = 0 if data["gender"] == "Male" else 1
    x[1] = float(data["age"])
    x[2] = float(data["hypertension"])
    x[3] = float(data["heart_disease"])
    x[4] = 0 if data["ever_married"] == "No" else 1
    x[5] = 0 if data["work_type"] == "Private" else \
        1 if data["work_type"] == "Self-emplyed" else \
            2 if data["work_type"] == "Never_worked" else \
                3 if data["work_type"] == "children" else 4
    x[6] = 0 if data["Residence_type"] else 1
    x[7] = float(data["avg_glucose_level"])
    x[8] = float(data["bmi"])
    x[9] = 0 if data["smoking_status"] == "never smoked" else \
        1 if data["smoking_status"] == "Unknown" or data["smoking_status"] == "formerly smoked" else 2

    result = model(x)

    result = 1 if result > 0.5 else 0
    print(result)
    return result

if __name__ == "__main__":
    # model = torch.jit.load("model/model/model_weights.pt")

    # onnx_model = onnx.load("model/model/model_weights.onnx")
    # model = ConvertModel(onnx_model)
    data = {
        "gender": "Male",
        "age": 67,
        "hypertension": 0,
        "heart_disease": 1,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "smoking_status": "formerly_smoked"
    }
    inference(data)


