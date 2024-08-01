import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, state_dim, num_uavs, action_dim):
        super(DQN, self).__init__()
        self.num_uavs = num_uavs
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_uavs * action_dim)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        if x.size(0) > 1:
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
        else:
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
        x = self.fc3(x)
        return x.view(-1, self.num_uavs, self.action_dim)


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, num_uavs, action_dim):
        super(DuelingDQN, self).__init__()
        self.num_uavs = num_uavs
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc_value = nn.Linear(256, 256)
        self.fc_advantage = nn.Linear(256, 256)
        
        self.value = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, num_uavs * action_dim)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        if x.size(0) > 1:
            x = F.relu(self.bn1(self.fc1(x)))
        else:
            x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        value = F.relu(self.fc_value(x))
        advantage = F.relu(self.fc_advantage(x))
        
        value = self.value(value)
        advantage = self.advantage(advantage)
        
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals.view(-1, self.num_uavs, self.action_dim)