import hydra
import torch
import gymnasium as gym
from itertools import count
from pathlib import Path
from tqdm import tqdm

import save
from agent import Agent

def train(cfg, device, output_dir):
    env = gym.make(cfg.env)
    agent = Agent(state_dim=env.observation_space.shape[0],
                  action_dim=env.action_space.shape[0],
                  min_action=env.action_space.low[0],
                  max_action=env.action_space.high[0],
                  device=device,
                  **cfg.agent).to(device)
    rewards = []
    pbar = tqdm(range(cfg.episodes), desc="Training")

    for e in pbar:
        state, info = env.reset()
        agent.reset()
        action = agent.get_action(state)

        # Loop until done
        while True:

            # Take a step.
            next_state, reward, terminated, truncated, info = env.step(action)
            next_action = agent.get_action(next_state)
            done = terminated or truncated

            # Update the agent
            agent.update(state=state, action=action, reward=reward, next_state=next_state, next_action=next_action, done=done)

            # Quit if done
            if done:
                break

            # For next iteration
            state = next_state
            action = next_action

        rewards.append(agent.cumulative_reward)
        pbar.set_postfix({"Cumulative Reward": f"{agent.cumulative_reward:.2f}"})
        pbar.update(1)

    env.close()  # Close the environment when done

    # Save
    save.plot(rewards=rewards, path=output_dir / "plot.png")
    save.save_rewards(rewards=rewards, path=output_dir / "rewards.pkl")
    agent.save_weights(output_dir=output_dir)


def infer(cfg, device, output_dir):
    env = gym.make(cfg.env, render_mode="rgb_array_list")
    agent = Agent(state_dim=env.observation_space.shape[0],
                  action_dim=env.action_space.shape[0],
                  min_action=env.action_space.low[0],
                  max_action=env.action_space.high[0],
                  device=device,
                  **cfg.agent).to(device)
    agent.load_weights(output_dir=output_dir)
    agent.eval()


    state, info = env.reset()
    agent.reset()

    for t in tqdm(range(cfg.infer_steps), desc="Inference"):
        action = agent.get_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            break

    images = env.render()
    save.save_video(frames=images, path=output_dir / "infer.mp4", fps=cfg.fps)
    env.close()

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    # Setup
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    output_dir = Path(cfg.output_dir) / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    train(cfg, device, output_dir)

    # Inference
    infer(cfg, device, output_dir)

if __name__ == "__main__":
    main()
