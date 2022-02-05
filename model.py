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
        state = torch.tensor(state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        new_state = torch.tensor(new_state, dtype=torch.float)

        # Handle multiple sizes:
        if len(state.shape) == 1:
            # shape should be (1, x) --> append 1 dimension
            state = torch.unsqueeze(state, dim=0)
            new_state = torch.unsqueeze(new_state, dim=0)
            action = torch.unsqueeze(action, dim=0)
            reward = torch.unsqueeze(reward, dim=0)

            # convert finish to a tuple
            finish = (finish, )

        # Predict Q value with current state
        pred = self.model(state)

        # Qnew = r + y(max(next_predQ))
        target = pred.clone()
        for idx in range(len(finish)):
            Qnew = reward[idx]
            if not finish[idx]:
                Qnew = reward[idx] + self.gamma*torch.max(self.model(new_state[idx]))
            
            target[idx][torch.argmax(action).item()] = Qnew
        
        # Optimizer
        self.optimizer.zero_grad()
        loss = self.loss(target, pred)
        loss.backward()
        self.optimizer.step()



