import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import numpy as np

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class StandardDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(StandardDQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
    


class DuelingDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
        
        self.value = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        feature = self.feature(x)
        advantage = self.advantage(feature)
        value = self.value(feature)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

class LSTMDQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, num_layers=2):
        super(LSTMDQN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        
    def forward(self, x):
        # Reshape input for LSTM if necessary
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        lstm_out, _ = self.lstm(x)
        x = F.relu(self.fc1(lstm_out[:, -1, :]))  # Take only the last output
        return self.fc2(x)
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Modified Agent class to support different architectures
class DQNAgents:
    def __init__(self, state_size, action_size, model_type='standard', 
                 device=get_device(),
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, 
                 epsilon_decay=0.995, learning_rate=0.001, 
                 batch_size=64, update_target_freq=10):
        
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.memory = ReplayBuffer(100000)
        self.model_type = model_type
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        self.train_step = 0
        
        # Initialize networks based on model type
        if model_type == 'standard':
            self.policy_net = StandardDQN(state_size, action_size).to(device)
            self.target_net = StandardDQN(state_size, action_size).to(device)
        elif model_type == 'dueling':
            self.policy_net = DuelingDQN(state_size, action_size).to(device)
            self.target_net = DuelingDQN(state_size, action_size).to(device)
        elif model_type == 'lstm':
            self.policy_net = LSTMDQN(state_size, action_size).to(device)
            self.target_net = LSTMDQN(state_size, action_size).to(device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
    
    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert lists to numpy arrays before creating tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        # Get current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values differently for Double DQN
        if self.model_type == 'double':
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
        else:
            next_q_values = self.target_net(next_states).max(1)[0]
        
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss and optimize
        loss = F.smooth_l1_loss(current_q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.train_step += 1
        if self.train_step % self.update_target_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()