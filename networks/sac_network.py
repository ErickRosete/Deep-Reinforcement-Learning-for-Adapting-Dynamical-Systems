import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from collections import OrderedDict
from utils.network import calculate_input_dim

def get_state_from_observation(tactile_network, state, detach_encoder):
    if isinstance(state, dict):
        fc_input = state["position"]
        if "force" in state:
            fc_input = torch.cat((fc_input, state["force"]), dim=-1)
        if "tactile_sensor" in state:
            tact_output = tactile_network(state["tactile_sensor"], detach_encoder)
            fc_input = torch.cat((fc_input, tact_output), dim=-1)
        return fc_input
    return state

class ActorNetwork(nn.Module):
    def __init__(self, tactile_network, observation_space, action_space, hidden_dim=256, tact_output=8):
        super(ActorNetwork, self).__init__()
        # Action parameters
        self.action_high = torch.tensor(action_space.high, dtype=torch.float, device="cuda") 
        self.action_low = torch.tensor(action_space.low, dtype=torch.float, device="cuda")
        action_dim = action_space.shape[0]

        self.tactile_network = tactile_network
        fc_input_dim = calculate_input_dim(observation_space, tact_output)

        self.fc_layers = nn.Sequential(OrderedDict([
                ('fc_1', nn.Linear(fc_input_dim, hidden_dim)),
                ('elu_1', nn.ELU()),
                ('fc_2', nn.Linear(hidden_dim, hidden_dim)),
                ('elu_2', nn.ELU())]))
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, observation, detach_encoder=False):
        state = get_state_from_observation(self.tactile_network, observation, detach_encoder)
        state = self.fc_layers(state)
        mean = self.fc_mean(state)
        log_std = self.fc_log_std(state)
        log_std = torch.clamp(log_std, min=-20, max=2) #Avoid -inf when std -> 0
        return mean, log_std

    def get_actions(self, state, deterministic=False, reparameterize=False, 
                    detach_encoder=False, epsilon=1e-6):

        mean, log_std = self.forward(state, detach_encoder)  
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

class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, tact_output=8):
        super(QNetwork, self).__init__()
        self.fc_layers = nn.Sequential(OrderedDict([
                ('fc_1', nn.Linear(input_dim, hidden_dim)),
                ('elu_1', nn.ELU()),
                ('fc_2', nn.Linear(hidden_dim, hidden_dim)),
                ('elu_2', nn.ELU()),
                ('fc_3', nn.Linear(hidden_dim, 1))]))

    def forward(self, state, action):
        fc_input = torch.cat((state, action), dim=-1)
        return self.fc_layers(fc_input)

class CriticNetwork(nn.Module):
    def __init__(self, tactile_network, observation_space, action_space, hidden_dim=256, tact_output=8):
        super(CriticNetwork, self).__init__()
        input_dim = action_space.shape[0]
        input_dim += calculate_input_dim(observation_space, tact_output)
        self.tactile_network = tactile_network
        self.Q1 = QNetwork(input_dim, hidden_dim, tact_output)
        self.Q2 = QNetwork(input_dim, hidden_dim, tact_output)

    def forward(self, observation, action, detach_encoder=False):
        state = get_state_from_observation(self.tactile_network, observation, detach_encoder)
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)
        return q1, q2