from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, name="model.pth"):
        model_folder = "./models"
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        file = os.path.join(model_folder,name)
        torch.save(self.state_dict(),file)
    
class QNetTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma=gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(),lr=self.lr)
        self.loss = nn.MSELoss()
    
    def train_step(self, state, action, reward, new_state, finish):
        pass


