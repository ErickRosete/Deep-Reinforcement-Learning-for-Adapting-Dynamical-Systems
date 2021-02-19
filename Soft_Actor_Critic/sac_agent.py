import os
import sys
import torch
import logging
import numpy as np
from torch import nn
from pathlib import Path
import torch.optim as optim
import torch.nn.functional as F
from pybulletX.utils.space_dict import SpaceDict
from torch.utils.tensorboard import SummaryWriter
sys.path.insert(0, str(Path(__file__).parents[1]))
from utils.network import transform_to_tensor
from Soft_Actor_Critic.replay_buffer import ReplayBuffer
from networks.sac_network import ActorNetwork, CriticNetwork
from networks.tactile_network import get_encoder_network, is_tactile_in_obs, TactileNetwork, DecoderNetwork

class SAC_Agent:
    def __init__(self, env, batch_size=256, gamma=0.99, tau=0.005, 
        actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, hidden_dim=256,
        tactile_dim=8, shared_encoder=True, ae_lr=3e-4):
        #Environment
        self.env = env
        self.with_tactile = is_tactile_in_obs(self.env.observation_space)
        self.shared_encoder = shared_encoder

        #Log 
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter()

        #Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        #Autoencoder
        self.decoder_latent_lambda = 0.0

        #Entropy
        self.alpha = 1
        self.target_entropy = -np.prod(env.action_space.shape).item()  # heuristic value
        self.log_alpha = torch.zeros(1, requires_grad=True, device="cuda")
        
        #Networks
        self.build_networks(hidden_dim, tactile_dim)
        self.build_optimizers(critic_lr, actor_lr, alpha_lr, ae_lr)

        self.loss_function = torch.nn.MSELoss()
        self.replay_buffer = ReplayBuffer()
        
    def build_networks(self, hidden_dim, tactile_dim=8):
        tactile_critic, tactile_critic_target, tactile_actor = nn.Identity(), nn.Identity(), nn.Identity()
        if self.with_tactile:
            #Share encoder in critic and actor
            encoder_network = get_encoder_network(self.env.observation_space["tactile_sensor"])
            tactile_critic = TactileNetwork(encoder_network, tactile_dim)
            if not self.shared_encoder:
                encoder_network = get_encoder_network(self.env.observation_space["tactile_sensor"])
            tactile_actor = TactileNetwork(encoder_network, tactile_dim)
            # Do not share encoder with target
            encoder_network = get_encoder_network(self.env.observation_space["tactile_sensor"])
            tactile_critic_target = TactileNetwork(encoder_network, tactile_dim)

            # Add autoencoder only for shared encoder
            if self.shared_encoder:
                self.decoder = DecoderNetwork(self.env.observation_space["tactile_sensor"], tactile_dim).cuda()

        input_dim = self.get_state_dim(self.env.observation_space, tactile_dim)
        self.actor = ActorNetwork(tactile_actor, input_dim, self.env.action_space, hidden_dim).cuda()
        
        input_dim += self.env.action_space.shape[0]
        self.critic = CriticNetwork(tactile_critic, input_dim, hidden_dim).cuda()
        self.critic_target = CriticNetwork(tactile_critic_target, input_dim, hidden_dim).cuda()
        self.critic_target.load_state_dict(self.critic.state_dict())

        
    @staticmethod
    def get_state_dim(observation_space, tact_output):
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

    def build_optimizers(self, critic_lr, actor_lr, alpha_lr, ae_lr=3e-4):
        if self.with_tactile and self.shared_encoder:
            ae_parameters = list(self.decoder.parameters()) + list(self.critic.tactile_network.parameters())
            self.autoencoder_optimizer = optim.Adam(ae_parameters, lr=ae_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    @staticmethod
    def soft_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_action_from_observation(self, observation, deterministic=False):
        """Interface to get action from SAC Actor, ready to be used in the environment"""
        observation = transform_to_tensor(observation)
        action, _ = self.actor.get_actions(observation, deterministic, reparameterize=False) 
        return action.detach().cpu().numpy()

    def get_states_from_observations(self, observations):
        if isinstance(observations, dict):
            states = observations["position"]
            if "force" in observations:
                states = torch.cat((states, observations["force"]), dim=-1)
            if "tactile_sensor" in observations:
                tact_output = self.tactile_network(observations["tactile_sensor"])
                states = torch.cat((states, tact_output), dim=-1)
            return states
        return observations

    def update_autoencoder(self, observations, log=True):
        EPS = 1e-5
        h = self.critic.tactile_network(observations, detach_encoder=False)
        pred_obs = self.decoder(h)
        # preprocess images to be in [-0.5, 0.5] range
        target_obs = observations / (observations.max() + EPS ) - 0.5
        rec_loss = F.mse_loss(pred_obs, target_obs.detach())

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        loss = rec_loss + self.decoder_latent_lambda * latent_loss
        self.autoencoder_optimizer.zero_grad()
        loss.backward()
        self.autoencoder_optimizer.step()

        if log:
            self.log_scalar('Train/Step/autoencoder_loss', loss.item(), self.training_step)
            
        return loss

    def update_critic(self, batch_observations, batch_actions, batch_next_observations, 
                     batch_rewards, batch_dones, log=True):

        with torch.no_grad():
            policy_actions, log_pi = self.actor.get_actions(batch_next_observations, deterministic=False,
                                    reparameterize=False)
            q1_next_target, q2_next_target = self.critic_target(batch_next_observations, policy_actions)
            q_next_target = torch.min(q1_next_target, q2_next_target)
            td_target = batch_rewards + (1 - batch_dones) * self.gamma * (q_next_target - self.alpha * log_pi)

        # Critic update
        q1, q2 = self.critic(batch_observations, batch_actions)
        critic_loss = self.loss_function(q1, td_target) + self.loss_function(q2, td_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if log:
            self.log_scalar('Train/Step/critic_loss', critic_loss.item(), self.training_step)

        return critic_loss.item()

    def update_actor_and_alpha(self, batch_observations, log=True):
        # Policy improvement (Do not update encoder network)
        policy_actions, log_pi = self.actor.get_actions(batch_observations, deterministic=False, 
                                    reparameterize=True, detach_encoder=self.shared_encoder)
        q1, q2 = self.critic(batch_observations, policy_actions, detach_encoder=self.shared_encoder)
        Q_value = torch.min(q1, q2)
        self.actor_optimizer.zero_grad()
        actor_loss = (self.alpha * log_pi - Q_value).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        #Update entropy parameter 
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        if log:
            self.log_scalar('Train/Step/actor_loss', actor_loss.item(), self.training_step)
            self.log_scalar('Train/Step/alpha_loss', alpha_loss.item(), self.training_step)

        return actor_loss.item(), alpha_loss.item()

    def update(self, observation, action, next_observation, reward, done, log):
        self.replay_buffer.add_transition(observation, action, next_observation, reward, done)

        # Sample next batch and perform batch update: 
        batch_observations, batch_actions, batch_next_observations, batch_rewards, batch_dones = \
            self.replay_buffer.next_batch(self.batch_size, tensor=True)

        critic_loss = self.update_critic(batch_observations, batch_actions, batch_next_observations, \
                                         batch_rewards, batch_dones, log)
        actor_loss, alpha_loss = self.update_actor_and_alpha(batch_observations, log)
        
        #Update target networks
        self.soft_update(self.critic_target, self.critic, self.tau)

        #Update decoder and encoder network
        if self.with_tactile and self.shared_encoder:
            self.update_autoencoder(batch_observations["tactile_sensor"], log)

        return critic_loss, actor_loss, alpha_loss

    def evaluate(self, num_episodes = 5, render=False):
        succesful_episodes, episodes_returns, episodes_lengths = 0, [], []
        for episode in range(1, num_episodes + 1):
            observation = self.env.reset()
            episode_return = 0
            for step in range(self.env.max_episode_steps):
                action = self.get_action_from_observation(observation, deterministic = True) 
                observation, reward, done, info = self.env.step(action)
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
        episode_return = 0
        ep_critic_loss, ep_actor_loss, ep_alpha_loss = 0, 0, 0
        observation = self.env.reset()
        for step in range(self.env.max_episode_steps):
            if episode < exploration_episodes:
                action = self.env.action_space.sample()
            else:
                action = self.get_action_from_observation(observation, deterministic = False) 
            next_observation, reward, done, info = self.env.step(action)
            critic_loss, actor_loss, alpha_loss = self.update(observation, action, next_observation,
                                                                reward, done, log)
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

    def train(self, num_episodes, exploration_episodes=0,
        log=True, eval_every=10, eval_episodes=5, render=False, early_stopping=False,
        save_dir="models/SAC_models", save_filename="sac_model", save_every=10): 

        self.training_step = 1
        best_eval_return = 0
        episodes_returns, episodes_lengths = [], []
        for episode in range(1, num_episodes + 1):
            episode_return, episode_length = self.train_episode(episode, exploration_episodes, log, render)
            episodes_returns.append(episode_return)
            episodes_lengths.append(episode_length)

            # Validation
            if episode % eval_every == 0 or episode == num_episodes:
                accuracy, eval_return, eval_length = self.evaluate(eval_episodes)
                self.log_episode_information(eval_return, eval_length, episode, "Validation", log)
                if eval_return > best_eval_return:
                    best_eval_return = eval_return
                    filename = self.get_custom_filename(save_dir, save_filename, "best_val")
                    self.save(filename)
                if accuracy == 1 and early_stopping:
                    self.logger.info("Early stopped as accuracy in validation is 1.0")
                    break

            # Save model
            if episode % save_every == 0:
                filename = self.get_save_filename(save_dir, save_filename, episode)
                self.save(filename)
        
        # End of training
        self.logger.info("Finished training after: %d steps", self.training_step)
        filename = self.get_save_filename(save_dir, save_filename, episode)
        self.save(filename)
        return episode, episodes_returns, episodes_lengths
    
    # Log methods
    def log_scalar(self, section, scalar, step):
            self.writer.add_scalar(section, scalar, step)  

    def log_episode_information(self, episode_return, episode_length, episode,
                                section="Train", tensorboard_log=True ):
        summary = "%s || Episode: %d   Return: %2f   Episode length: %d" % (section, 
                  episode, episode_return, episode_length)
        self.logger.info(summary)
        if tensorboard_log:
            self.writer.add_scalar('%s/Episode/return' % section, episode_return, episode)
            self.writer.add_scalar('%s/Episode/length' % section, episode_length, episode)

    @staticmethod
    def get_save_filename(save_dir, save_filename, episode):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)           
        filename = "%s/%s_%d.pth"%(save_dir, save_filename, episode)
        return filename

    @staticmethod
    def get_custom_filename(save_dir, save_filename, text):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)           
        filename = "%s/%s_%s.pth"%(save_dir, save_filename, text)
        return filename
        
    def save(self, filename):
        checkpoint ={ 'actor_dict': self.actor.state_dict(),
                      'critic_dict' : self.critic.state_dict()  }
        torch.save(checkpoint, filename)

    def load(self, filename):
        if os.path.isfile(filename):
            print("=> loading checkpoint... ")
            checkpoint = torch.load(filename)
            self.actor.load_state_dict(checkpoint['actor_dict'])
            self.critic.load_state_dict(checkpoint['critic_dict'])
            print("done !")
        else:
            print("no checkpoint found...")

    
