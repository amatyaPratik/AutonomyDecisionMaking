# Import:
# -------
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Deep Q-Network:
# ---------------
class Qnet(nn.Module):
    def __init__(self, no_actions, no_states):
        super(Qnet, self).__init__()
        # no_states = np.array(no_states).reshape(1, -1)
        self.model = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(no_states, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, no_actions)
        )
        self.apply(self.weights_init)  # Apply weight initialization

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)

    def sample_action(self, observation, epsilon):
        a = self.forward(observation)

        #! Exploration
        if random.random() < epsilon:
            return random.randint(0, self.model[-1].out_features - 1)

        #! Exploitation
        else:
            return a.argmax().item()
