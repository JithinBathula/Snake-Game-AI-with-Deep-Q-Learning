import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import os

class Linear_QNet(nn.Module):
    def __init__(self, inputsize, hiddensize,outputsize):
        super(Linear_QNet, self).__init__()
        self.ln1 = nn.Linear(inputsize,hiddensize)
        self.ln2 = nn.Linear(hiddensize,outputsize)
        

    def forward(self,x):
        x = self.ln1(x)
        x = F.relu(x)
        x = self.ln2(x)
        return x
    def save_model(self, file_name = "model.pth"):
        model_folder = "./model"
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        
        file_name = os.path.join(model_folder,file_name)
        torch.save(self.state_dict(),file_name)

class QTrainer:
    def __init__(self, model, learningrate, gamma):
        self.learninrate = learningrate
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), learningrate)
        self.lossfunc = nn.MSELoss()

    def train_step(self,state,action,reward,next_state,done):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            next_state = torch.unsqueeze(next_state,0)
            done = (done,)

        pred = self.model(state)

        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.lossfunc(target,pred)
        loss.backward()
        self.optimizer.step()


