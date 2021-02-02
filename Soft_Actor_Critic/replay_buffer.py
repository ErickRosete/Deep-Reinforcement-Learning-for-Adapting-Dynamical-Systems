from collections import namedtuple
import numpy as np
import torch
from torch import tensor

class ReplayBuffer:
    def __init__(self, max_capacity=1e6): 
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "dones"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], dones=[])
        self.max_capacity = max_capacity
    
    def current_capacity(self):
        return len(self._data.states)

    def add_transition(self, state, action, next_state, reward, done):
        """
        This method adds a transition to the replay buffer.
        """
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.dones.append(done)

        if len(self._data.states) > self.max_capacity:
            del self._data.states[0]
            del self._data.actions[0]
            del self._data.next_states[0]
            del self._data.rewards[0]
            del self._data.dones[0]

    def transform_to_batch(self, array, batch_indices):
        if isinstance(array[0], dict):
            batch = {}
            for key in array[0].keys():
                batch[key] = np.array([array[i][key] for i in batch_indices])
        else:
            batch = np.array([array[i] for i in batch_indices])
        return batch

    def transform_to_tensor(self, x, dtype=torch.float, grad=True):
        if isinstance(x, dict):
            tensor = {k: torch.tensor(v, dtype=dtype, device="cuda", requires_grad=grad) for k, v in x.items()}
        else:
            tensor = torch.tensor(x, dtype=dtype, device="cuda", requires_grad=grad) #B,S_D
        return tensor

    def next_batch(self, batch_size, tensor=False):
        """
        This method samples a batch of transitions.
        """
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = self.transform_to_batch(self._data.states, batch_indices)
        batch_actions = self.transform_to_batch(self._data.actions, batch_indices)
        batch_next_states = self.transform_to_batch(self._data.next_states, batch_indices)
        batch_rewards = np.expand_dims(self.transform_to_batch(self._data.rewards, batch_indices), axis=-1)
        batch_dones = np.expand_dims(self.transform_to_batch(self._data.dones, batch_indices), axis=-1)
        
        if tensor:        
            batch_states = self.transform_to_tensor(batch_states) #B,S_D
            batch_actions = self.transform_to_tensor(batch_actions) #B,A_D
            batch_next_states = self.transform_to_tensor(batch_next_states, grad=False)
            batch_rewards = self.transform_to_tensor(batch_rewards, grad=False) #B,1
            batch_dones = self.transform_to_tensor(batch_dones, dtype=torch.uint8, grad=False) #B,1
      
        return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones
