import torch
reward = 10
reward = torch.tensor(reward, dtype=torch.float)
print(reward)
reward = torch.unsqueeze(reward, 0)
print(reward)