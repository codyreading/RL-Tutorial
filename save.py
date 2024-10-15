import pickle
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

def plot(rewards, path):
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumalative Reward')
    plt.savefig(path)

def save_rewards(rewards, path):
    with open(path, "wb") as file:
        pickle.dump(rewards, file)

def save_video(frames, path, fps):
    # Create a video clip from the frames
    clip = ImageSequenceClip(frames, fps=fps)

    # Write the video file to the output path
    clip.write_videofile(str(path), codec='libx264')