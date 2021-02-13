import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from collections import OrderedDict
from pybulletX.utils.space_dict import SpaceDict

""" Common methods between Actor and Critic """
def get_tactile_network(observation_space, tact_output):
    tactile_network = None
    if isinstance(observation_space, SpaceDict):
        keys = list(observation_space.keys())
        if "tactile_sensor" in keys:
            tactile_network = TactileNetwork(observation_space["tactile_sensor"], tact_output)
    return tactile_network

def calculate_input_dim(observation_space, tact_output):
    if isinstance(observation_space, SpaceDict):
        fc_input_dim = 0
        keys = list(observation_space.keys())
        if "force" in keys:
            fc_input_dim += 2
        if "position" in keys:
            fc_input_dim += 3
        if "tactile_sensor" in keys:
            fc_input_dim += tact_output
        return fc_input_dim
    return observation_space.shape[0]

def get_fc_input(tactile_network, state):
    if isinstance(state, dict):
        fc_input = state["position"]
        if "force" in state:
            fc_input = torch.cat((fc_input, state["force"]), dim=-1)
        if "tactile_sensor" in state:
            tact_output = tactile_network(state["tactile_sensor"])
            fc_input = torch.cat((fc_input, tact_output), dim=-1)
        return fc_input
    return state

class TactileNetwork(nn.Module):
    def __init__(self, observation_space, output_dim):
        super(TactileNetwork, self).__init__()
        h, w = observation_space[0].shape
        h, w = self.calc_out_size(h, w, 8, 0, 4)
        h, w = self.calc_out_size(h, w, 4, 0, 2)

        self.network = nn.Sequential(OrderedDict([
            ('tact_cnn_1', nn.Conv2d(1, 16, 8, stride=4)),
            ('tact_cnn_elu_1', nn.ELU()),
            ('tact_cnn_2', nn.Conv2d(16, 32, 4, stride=2)),
            ('spatial_softmax', SpatialSoftmax(h, w)), #Batch_size, 2 * num_channels
        ]))
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        s1 = self.network(x[:,0].unsqueeze(1)).squeeze()
        s2 = self.network(x[:,1].unsqueeze(1)).squeeze()
        output = torch.cat((s1, s2), dim=-1)
        output = self.fc(output)
        return output

    @staticmethod
    def calc_out_size(w, h, kernel_size, padding, stride):
        width = (w - kernel_size + 2 * padding)//stride + 1
        height = (h - kernel_size + 2 * padding)//stride + 1
        return width, height

class SpatialSoftmax(nn.Module):
    # reference: https://arxiv.org/pdf/1509.06113.pdf
    def __init__(self, height, width):
        super(SpatialSoftmax, self).__init__()
        x_map = np.empty([height, width], np.float32)
        y_map = np.empty([height, width], np.float32)

        for i in range(height):
            for j in range(width):
                x_map[i, j] = (i - height / 2.0) / height
                y_map[i, j] = (j - width / 2.0) / width

        self.x_map = torch.from_numpy(np.array(x_map.reshape((-1)), np.float32)).cuda() # W*H
        self.y_map = torch.from_numpy(np.array(x_map.reshape((-1)), np.float32)).cuda() # W*H

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]) # batch, C, W*H
        x = F.softmax(x, dim=2) # batch, C, W*H
        fp_x = torch.matmul(x, self.x_map) # batch, C
        fp_y = torch.matmul(x, self.y_map) # batch, C
        x = torch.cat((fp_x, fp_y), 1)
        return x # batch, C*2

class ActorNetwork(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dim=256, tact_output=8):
        super(ActorNetwork, self).__init__()
        # Action parameters
        self.action_high = torch.tensor(action_space.high, dtype=torch.float, device="cuda") 
        self.action_low = torch.tensor(action_space.low, dtype=torch.float, device="cuda")
        action_dim = action_space.shape[0]

        self.tactile_network = get_tactile_network(observation_space, tact_output)
        fc_input_dim = calculate_input_dim(observation_space, tact_output)

        self.fc_layers = nn.Sequential(OrderedDict([
                ('fc_1', nn.Linear(fc_input_dim, hidden_dim)),
                ('elu_1', nn.ELU()),
                ('fc_2', nn.Linear(hidden_dim, hidden_dim)),
                ('elu_2', nn.ELU())]))
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        fc_input = get_fc_input(self.tactile_network, state)
        fc_output = self.fc_layers(fc_input)
        mean = self.fc_mean(fc_output)
        log_std = self.fc_log_std(fc_output)
        log_std = torch.clamp(log_std, min=-20, max=2) #Avoid -inf when std -> 0
        return mean, log_std

    def get_actions(self, state, deterministic=False, reparameterize=False, epsilon=1e-6):
        mean, log_std = self.forward(state)  
        if deterministic:
            actions = torch.tanh(mean)
            log_pi = torch.zeros_like(actions)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            if reparameterize:
                z = normal.rsample()
            else:
                z = normal.sample()
            actions = torch.tanh(z)
            log_pi = normal.log_prob(z) - torch.log(1 - actions.square() + epsilon)
            log_pi = log_pi.sum(-1, keepdim=True)
        return self.scale_actions(actions), log_pi 

    def scale_actions(self, action):
        slope = (self.action_high - self.action_low) / 2
        action = self.action_low + slope * (action + 1)
        return action

class CriticNetwork(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dim=256, tact_output=8):
        super(CriticNetwork, self).__init__()
        # Sawyer Peg Env Network
        self.tactile_network = get_tactile_network(observation_space, tact_output)
        fc_input_dim = action_space.shape[0]
        fc_input_dim += calculate_input_dim(observation_space, tact_output)

        self.fc_layers = nn.Sequential(OrderedDict([
                ('fc_1', nn.Linear(fc_input_dim, hidden_dim)),
                ('elu_1', nn.ELU()),
                ('fc_2', nn.Linear(hidden_dim, hidden_dim)),
                ('elu_2', nn.ELU()),
                ('fc_3', nn.Linear(hidden_dim, 1))]))

    def forward(self, state, action):
        fc_input = get_fc_input(self.tactile_network, state)
        fc_input = torch.cat((fc_input, action), dim=-1)
        return self.fc_layers(fc_input)

