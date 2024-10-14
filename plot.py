import matplotlib.pyplot as plt

def plot(rewards, name):
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumalative Reward')
    plt.savefig(f"{name}.png")
