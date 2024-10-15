import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import Actor, Critic
from buffer import Buffer
from ou_noise import OUNoise

class Agent(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 min_action,
                 max_action,
                 seed,
                 lr_actor,
                 lr_critic,
                 gamma,
                 tau,
                 fc1_dim,
                 fc2_dim,
                 buffer_size,
                 batch_size,
                 on_policy,
                 replay_buffer,
                 target_networks,
                 noise,
                 device):
        super().__init__()
        self.min_action = min_action
        self.max_action = max_action
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.tau = tau
        self.on_policy = on_policy
        self.replay_buffer = replay_buffer
        self.target_networks = target_networks
        self.device = device

        # Setup networks and optimizers
        self.actor = Actor(state_dim=state_dim, action_dim=action_dim, fc1_dim=fc1_dim, fc2_dim=fc2_dim, seed=seed)
        self.critic = Critic(state_dim=state_dim, action_dim=action_dim, fc1_dim=fc1_dim, fc2_dim=fc2_dim, seed=seed)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)


        if self.target_networks:
            self.actor_target = Actor(state_dim=state_dim, action_dim=action_dim, fc1_dim=fc1_dim, fc2_dim=fc2_dim, seed=seed)
            self.critic_target = Critic(state_dim=state_dim, action_dim=action_dim, fc1_dim=fc1_dim, fc2_dim=fc2_dim, seed=seed)
        else:
            self.actor_target = self.actor
            self.critic_target = self.critic

        # Setup large buffer is using replay buffer
        if replay_buffer:
            self.buffer_size = buffer_size
            self.batch_size = batch_size

        # Only keep the last batch elements otherwise
        else:
            self.buffer_size = batch_size

        # Create buffer
        self.buffer = Buffer(buffer_size=self.buffer_size, device=device, seed=seed)

        # Noise process
        self.noise = OUNoise(size=action_dim, seed=seed, device=device, **noise)

        # Reset
        self.reset()

    def load_weights(self, output_dir):
        self.actor.load_state_dict(torch.load(output_dir / 'actor.pth', weights_only=True))
        self.critic.load_state_dict(torch.load(output_dir / 'critic.pth', weights_only=True))

    def save_weights(self, output_dir):
        torch.save(self.actor.state_dict(), output_dir / 'actor.pth')
        torch.save(self.critic.state_dict(), output_dir / 'critic.pth')

    def reset(self):
        self.cumulative_reward = 0.0

    def get_action(self, state):
        state_tensor = torch.as_tensor(state).to(self.device)

        # Get action
        with torch.no_grad():
            if self.training:
                self.actor.eval()
                action = self.actor(state_tensor)
                self.actor.train()
                action += self.noise.sample()
            else:
                action = self.actor(state_tensor)

        # Get action compatible with env
        action = action.detach().cpu().numpy()
        action *= self.max_action # Scale to [-2, 2]
        action = np.clip(action, self.min_action, self.max_action)
        return action

    def update(self, state, action, reward, next_state, next_action, done):

        # Store data
        self.buffer.add(state=state,
                        action=action / self.max_action,
                        reward=reward,
                        next_state=next_state,
                        next_action=next_action / self.max_action,
                        done=done)
        self.cumulative_reward += reward

        if not self.replay_buffer and self.buffer.full():
            batch = self.buffer.pop()
            self.optimize_step(**batch)
        elif self.replay_buffer and len(self.buffer) > self.batch_size:
            batch = self.buffer.sample(self.batch_size)
            self.optimize_step(**batch)

    def optimize_step(self, state, action, reward, next_state, next_action, done):
        # Compute target
        with torch.no_grad():
            if not self.on_policy:
                next_action = self.actor_target(next_state)
            target_q = reward + self.gamma * (1 - done) * self.critic_target(state=next_state, action=next_action)

        # Critic update
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Actor update
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Soft update target networks
        if self.target_networks:
            self.actor_target = self.soft_update(source=self.actor, target=self.actor_target)
            self.critic_target = self.soft_update(source=self.critic, target=self.critic_target)

    def soft_update(self, source, target):
        """Softly copy weights"""
        source_state_dict = source.state_dict()
        target_state_dict = target.state_dict()

        for key in source_state_dict.keys():
            target_state_dict[key] = self.tau * source_state_dict[key] + (1 - self.tau) * target_state_dict[key]

        target.load_state_dict(target_state_dict)
        return target
