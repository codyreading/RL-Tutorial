import torch
import torch.nn as nn
import torch.optim as optim

from model import Actor, Critic

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
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device

        # Setup networks and optimizers
        self.actor = Actor(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.critic = Critic(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        # Reset
        self.reset()

    def reset(self):
        self.cumulative_reward = 0.0

    def get_action(self, state):
        # Get action
        state_tensor = torch.as_tensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state_tensor)

        # Add noise
        if self.training:
            noise = self.noise_scale * torch.randn_like(action)
            action = action + noise
            action = torch.clamp(action, self.min_action, self.max_action)

        # Get numpy action
        action = action.detach().cpu().numpy()[0]
        return action