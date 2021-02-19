import sys
import torch
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from Soft_Actor_Critic.sac_agent import SAC_Agent

# TODO: Consider the idea of backpropagating gmm instead of seeing the gmm action as 
# part of the state
class SAC_GMM_Residual_Agent(SAC_Agent):
    def __init__(self, model, *args, **kwargs):
        self.model = model
        self.action_in_obs = False
        self.burn_in_steps = 10000
        super(SAC_GMM_Residual_Agent, self).__init__(*args, **kwargs)

    def get_state_dim(self, observation_space, tact_output):
        state_dim = super(SAC_GMM_Residual_Agent, self).get_state_dim(observation_space, tact_output)
        if self.action_in_obs:
            state_dim += self.env.action_space.shape[0] # We add gmm action to the state dim
        return state_dim

    def evaluate(self, num_episodes = 5, render=False):
        # TODO: Think of a nicer way to reuse code
        succesful_episodes, episodes_returns, episodes_lengths = 0, [], []
        for episode in range(1, num_episodes + 1):
            observation = self.env.reset()
            # === Start of modified code ===
            if self.action_in_obs:
                observation['gmm_action'] = self.model.predict_velocity_from_observation(observation)
            episode_return = 0
            for step in range(self.env.max_episode_steps):
                residual_action = self.get_action_from_observation(observation, deterministic = True) 
                env_action = residual_action + self.model.predict_velocity_from_observation(observation)
                env_action = np.clip(env_action, -1, 1)
                observation, reward, done, info = self.env.step(env_action)
                if self.action_in_obs:
                    observation['gmm_action'] = self.model.predict_velocity_from_observation(observation)
                # === End of modified code ===
                episode_return += reward
                if render:
                    self.env.render()
                if done:
                    break
            if ("success" in info) and info['success']:
                succesful_episodes += 1
            episodes_returns.append(episode_return) 
            episodes_lengths.append(step)
        accuracy = succesful_episodes/num_episodes
        return accuracy, np.mean(episodes_returns), np.mean(episodes_lengths)

    def train_episode(self, episode, exploration_episodes, log, render):
        # TODO: Think of a nicer way to reuse code
        episode_return = 0
        ep_critic_loss, ep_actor_loss, ep_alpha_loss = 0, 0, 0
        #  === Start of modified code ===
        observation = self.env.reset()
        if self.action_in_obs:
            observation['gmm_action'] = self.model.predict_velocity_from_observation(observation)
        for step in range(self.env.max_episode_steps):
            if self.training_step < self.burn_in_steps:
                residual_action = np.zeros(self.env.action_space.shape)
            else:
                residual_action = self.get_action_from_observation(observation, deterministic = False)
            env_action = residual_action + self.model.predict_velocity_from_observation(observation)
            env_action = np.clip(env_action, -1, 1)
            next_observation, reward, done, info = self.env.step(env_action)
            if self.action_in_obs:
                next_observation['gmm_action'] = self.model.predict_velocity_from_observation(next_observation)
            critic_loss, actor_loss, alpha_loss = self.update(observation, residual_action, next_observation,
                                                                reward, done, log)
            # === End of modified code ===
            observation = next_observation
            episode_return += reward
            ep_critic_loss += critic_loss
            ep_actor_loss += actor_loss
            ep_alpha_loss += alpha_loss
            self.training_step += 1

            if render:
                self.env.render()
            if done:
                break
        episode_length = step

        if log:
            self.log_scalar('Train/Episode/critic_loss', ep_critic_loss/episode_length, episode)
            self.log_scalar('Train/Episode/actor_loss', ep_actor_loss/episode_length, episode)
            self.log_scalar('Train/Episode/alpha_loss', ep_alpha_loss/episode_length, episode)
            self.log_episode_information(episode_return, episode_length, episode, "Train")
        return episode_return, episode_length