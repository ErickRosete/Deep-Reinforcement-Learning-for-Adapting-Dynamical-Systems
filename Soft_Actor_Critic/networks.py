import torch
import torch.nn as nn
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
    def __init__(self, observation_space, output_dim=8):
        super(TactileNetwork, self).__init__()
        in_channels = len(observation_space)
        h, w = observation_space[0].shape
        h, w = self.calc_out_size(h, w, 8, 0, 4)
        h, w = self.calc_out_size(h, w, 4, 0, 2)
        h, w = self.calc_out_size(h, w, 3, 0, 1)

        self.conv_layers = nn.Sequential(OrderedDict([
            ('tact_cnn_1', nn.Conv2d(in_channels, 32, 8, stride=4)),
            ('tact_cnn_relu_1', nn.ReLU()),
            ('tact_cnn_2', nn.Conv2d(32, 64, 4, stride=2)),
            ('tact_cnn_relu_2', nn.ReLU()),
            ('tact_cnn_3', nn.Conv2d(64, 32, 3, stride=1)),
            ('tact_cnn_relu_3', nn.ReLU()),
        ]))

        self.fc_layers = nn.Sequential(OrderedDict([
            ('tact_fc_1', nn.Linear(h * w * 32, output_dim)),
            ('tact_fc_relu_1', nn.ReLU())
        ]))

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        x = self.conv_layers(x)
        x = x.view(batch_size, -1).squeeze()
        return self.fc_layers(x)

    @staticmethod
    def calc_out_size(w, h, kernel_size, padding, stride):
        width = (w - kernel_size + 2 * padding)//stride + 1
        height = (h - kernel_size + 2 * padding)//stride + 1
        return width, height

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
                ('relu_1', nn.ReLU()),
                ('fc_2', nn.Linear(hidden_dim, hidden_dim)),
                ('relu_2', nn.ReLU())]))
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
                ('relu_1', nn.ReLU()),
                ('fc_2', nn.Linear(hidden_dim, hidden_dim)),
                ('relu_2', nn.ReLU()),
                ('fc_3', nn.Linear(hidden_dim, 1))]))

    def forward(self, state, action):
        fc_input = get_fc_input(self.tactile_network, state)
        fc_input = torch.cat((fc_input, action), dim=-1)
        return self.fc_layers(fc_input)
