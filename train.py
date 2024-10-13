import hydra
import torch
import gymnasium as gym
from itertools import count

from tqdm import tqdm
from agent import Agent
from plot import plot

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):

    # Setup
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    env = gym.make(cfg.env, render_mode=cfg.render_mode)
    agent = Agent(state_dim=env.observation_space.shape[0],
                  action_dim=env.action_space.shape[0],
                  min_action=env.action_space.low[0],
                  max_action=env.action_space.high[0],
                  device=device,
                  **cfg.agent).to(device)
    rewards = []


    for e in tqdm(range(cfg.episodes), desc="Episodes"):
        state, info = env.reset()
        agent.reset()

        # Loop until done
        for s in count():

            # Take a step
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Update the agent
            agent.update(state=state, reward=reward)

            # Quit if done
            if done:
                break


        cumulative_reward = agent.get_cumulative_reward()
        rewards.append(cumulative_reward)

    plot(rewards=rewards, name="part1")
    env.close()  # Close the environment when done



if __name__ == "__main__":
    main()
