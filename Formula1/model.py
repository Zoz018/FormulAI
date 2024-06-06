import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear_dir = nn.Linear(hidden_size, output_size)  # Output layer for direction
        self.linear_acc = nn.Linear(hidden_size, output_size)  # Output layer for acceleration

    def forward(self, x):
        x = F.relu(self.linear1(x))
        dir_output = F.sigmoid(self.linear_dir(x))  # Direction output
        acc_output = F.sigmoid(self.linear_acc(x))  # Acceleration output
        return dir_output, acc_output

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

    def train_step(self, state, action_dir, action_acc, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action_dir = torch.tensor(action_dir, dtype=torch.long)  # Convert to tensor
        action_acc = torch.tensor(action_acc, dtype=torch.long)  # Convert to tensor
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.bool)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action_dir = torch.unsqueeze(action_dir, 0)
            action_acc = torch.unsqueeze(action_acc, 0)
            reward = torch.unsqueeze(reward, 0)
            done = torch.unsqueeze(done, 0)

        # Predicted Q values with current state
        pred_dir, pred_acc = self.model(state)

        # Compute Q_new for both direction and acceleration
        target_dir = pred_dir.clone()
        target_acc = pred_acc.clone()
        for idx in range(len(done)):
            Q_new_dir = reward[idx]
            Q_new_acc = reward[idx]
            if not done[idx]:
                next_pred_dir, next_pred_acc = self.model(state)
                Q_new_dir = reward[idx] + self.gamma * torch.max(next_pred_dir)
                Q_new_acc = reward[idx] + self.gamma * torch.max(next_pred_acc)

            target_dir[idx][torch.argmax(action_dir[idx])] = Q_new_dir
            target_acc[idx][torch.argmax(action_acc[idx])] = Q_new_acc
            
        # Compute loss and perform backpropagation for both direction and acceleration
        self.optimizer.zero_grad()
        loss_dir = self.criterion(pred_dir, target_dir)
        loss_acc = self.criterion(pred_acc, target_acc)
        loss = loss_dir + loss_acc
        loss.backward()
        self.optimizer.step()