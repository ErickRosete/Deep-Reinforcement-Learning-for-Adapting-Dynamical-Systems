import sys
import gym
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from GMM.gmm import GMM
from Soft_Actor_Critic.sac_agent import SAC_Agent

class SAC_GMM_Agent(SAC_Agent):
    def __init__(self, model, window_size=32, *args, **kwargs):
        self.initial_model = model      # Initial model provided
        self.model = GMM() 
        self.model.copy_model(self.initial_model)    # Model used for training
        self.window_size = window_size
        self.burn_in_steps = 1000
        super(SAC_GMM_Agent, self).__init__(*args, **kwargs)

    def get_action_space(self):
        if not hasattr(self, 'action_space'):
            priors_high = np.ones(self.model.priors.size) * 0.1
            mu_high = np.ones(self.model.mu.size) * 0.005
            action_high = np.concatenate((priors_high, mu_high), axis=-1)
            action_low = - action_high
            self.action_space = gym.spaces.Box(action_low, action_high)
        return self.action_space

    def update_gaussians(self, gmm_change):
        priors = gmm_change[:self.model.priors.size]
        priors = priors.reshape(self.model.priors.shape)
        mu = gmm_change[self.model.priors.size:]
        mu = mu.reshape(self.model.mu.shape)
        change_dict = {"mu":mu, "prior":priors}
        self.model.update_gaussians(change_dict)

    def evaluate(self, num_episodes = 5, render=False):
        succesful_episodes, episodes_returns, episodes_lengths = 0, [], []
        for episode in range(1, num_episodes + 1):
            observation = self.env.reset()
            episode_return, episode_length, left_steps = 0, 0, self.env.max_episode_steps
            while left_steps > 0:
                self.model.copy_model(self.initial_model)
                gmm_change = self.get_action_from_observation(observation, deterministic = True)
                self.update_gaussians(gmm_change)
                model_reward = 0
                for step in range(self.window_size): 
                    vel = self.model.predict_velocity_from_observation(observation)
                    observation, reward, done, info = self.env.step(vel)
                    model_reward += reward
                    episode_length += 1
                    left_steps -= 1
                    if render:
                        self.env.render()
                    if done or left_steps <= 0:
                        break
                episode_return += model_reward
                if done:
                    break
                episode_return += reward
                if render:
                    self.env.render()
                if done:
                    break
            if ("success" in info) and info['success']:
                succesful_episodes += 1
            episodes_returns.append(episode_return) 
            episodes_lengths.append(episode_length)
        accuracy = succesful_episodes/num_episodes
        return accuracy, np.mean(episodes_returns), np.mean(episodes_lengths)

    def train_episode(self, episode, exploration_episodes, log, render):
        sac_steps = 0
        episode_return, episode_length, left_steps = 0, 0, self.env.max_episode_steps
        ep_critic_loss, ep_actor_loss, ep_alpha_loss = 0, 0, 0
        observation = self.env.reset()
        while left_steps > 0:
            self.model.copy_model(self.initial_model)
            if self.training_step < self.burn_in_steps:
                gmm_change = np.zeros(self.action_space.shape)
            else:
                gmm_change = self.get_action_from_observation(observation, deterministic = False)
            self.update_gaussians(gmm_change)
            model_reward = 0
            curr_observation = observation
            for step in range(self.window_size): 
                vel = self.model.predict_velocity_from_observation(curr_observation)
                curr_observation, reward, done, info = self.env.step(vel)
                model_reward += reward
                episode_length += 1
                left_steps -= 1
                if render:
                    self.env.render()
                if done or left_steps <= 0:
                    break
            episode_return += model_reward
            if done:
                break
            critic_loss, actor_loss, alpha_loss = self.update(observation, gmm_change, curr_observation,
                                                                reward, done, log)
            observation = curr_observation
            episode_return += reward
            ep_critic_loss += critic_loss
            ep_actor_loss += actor_loss
            ep_alpha_loss += alpha_loss
            self.training_step += 1 # SAC_Steps in total
            sac_steps += 1 # SAC_Steps in this episode

            if render:
                self.env.render()
            if done:
                break

        if log:
            self.log_scalar('Train/Episode/critic_loss', ep_critic_loss/sac_steps, episode)
            self.log_scalar('Train/Episode/actor_loss', ep_actor_loss/sac_steps, episode)
            self.log_scalar('Train/Episode/alpha_loss', ep_alpha_loss/sac_steps, episode)
            self.log_episode_information(episode_return, episode_length, episode, "Train")

        return episode_return, episode_length