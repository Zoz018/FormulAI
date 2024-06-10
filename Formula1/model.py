import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        # self.linear_dir = nn.Linear(hidden_size, output_size)  # Output layer for direction
        # self.linear_acc = nn.Linear(hidden_size, output_size)  # Output layer for acceleration

    def forward(self, x):
        x = F.relu(self.linear1(x))
        output = F.sigmoid(self.linear2(x))
        # dir_output = F.sigmoid(self.linear_dir(x))  # Direction output
        # acc_output = F.sigmoid(self.linear_acc(x))  # Acceleration output
        # return dir_output, acc_output
        return output

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)  # Convert to tensor
        reward = torch.tensor(reward, dtype=torch.int)
        done = torch.tensor(done, dtype=torch.bool)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = torch.unsqueeze(done, 0)

        # Predicted Q values with current state
        pred = self.model(state)

        # Compute Q_new for both direction and acceleration
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                next_pred = self.model(next_state[idx])
                Q_new = reward[idx] + self.gamma * torch.max(next_pred)
              

            target[idx][torch.argmax(action[idx])] = Q_new

        # Compute loss and perform backpropagation for both direction and acceleration
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()