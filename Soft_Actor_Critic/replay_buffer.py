import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import torch
import numpy as np
from utils.network import transform_to_tensor
from collections import namedtuple

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

    @staticmethod
    def transform_to_batch(array, batch_indices):
        if isinstance(array[0], dict):
            batch = {}
            for key in array[0].keys():
                batch[key] = np.array([array[i][key] for i in batch_indices])
        else:
            batch = np.array([array[i] for i in batch_indices])
        return batch

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
            batch_states = transform_to_tensor(batch_states) #B,S_D
            batch_actions = transform_to_tensor(batch_actions) #B,A_D
            batch_next_states = transform_to_tensor(batch_next_states, grad=False)
            batch_rewards = transform_to_tensor(batch_rewards, grad=False) #B,1
            batch_dones = transform_to_tensor(batch_dones, dtype=torch.uint8, grad=False) #B,1
      
        return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones
