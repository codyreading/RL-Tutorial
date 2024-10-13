import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import Actor, Critic
from buffer import Buffer

class Agent(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 min_action,
                 max_action,
                 lr,
                 gamma,
                 noise_scale,
                 hidden_dim,
                 buffer_size,
                 batch_size,
                 on_policy,
                 replay_buffer,
                 device):
        super().__init__()
        self.min_action = min_action
        self.max_action = max_action
        self.noise_scale = noise_scale
        self.replay_buffer = replay_buffer
        self.device = device

        # Setup networks and optimizers
        self.actor = Actor(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.actor_target = Actor(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.critic = Critic(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.critic_target = Critic(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        # Setup large buffer is using replay buffer
        if replay_buffer:
            self.buffer_size = buffer_size

        # Only keep the last batch elements otherwise
        else:
            self.buffer_size = batch_size

        # Create buffer
        self.buffer = Buffer(self.buffer_size)

        # Reset
        self.reset()

    def reset(self):
        self.cumulative_reward = 0.0
        self.buffer.clear()

    def get_action(self, state):
        # Get action
        state_tensor = torch.as_tensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state_tensor)

        # Add noise
        if self.training:
            noise = self.noise_scale * torch.randn_like(action)
            action = action + noise

        # Get action compatible with env
        action = action.detach().cpu().numpy()[0]
        action *= self.max_action # Scale to [-2, 2]
        action = np.clip(action, self.min_action, self.max_action)
        return action

    def update(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)
        self.cumulative_reward += reward

        if not self.replay_buffer and self.buffer.full():
            breakpoint()
            # TODO Get batch from buffer
            self.optimize_step(state, action, reward, next_state, done)

    def optimize_step(self, state, action, reward, next_state, done, next_action=None):
        # Compute target
        with torch.no_grad():
            if not self.on_policy:
                next_action = self.actor_target(next_state)
            target_q = reward + self.gamma * (1 - done) * self.critic(state=next_state, action=next_action)

        # Critic update
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        # Actor update
        actor_loss = -self.critic(state, self.actor(state)).mean()
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        # Soft update target networks
