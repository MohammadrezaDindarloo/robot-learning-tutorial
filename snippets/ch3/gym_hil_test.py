import gymnasium as gym
import gym_hil
import matplotlib.pyplot as plt

def main():
    env = gym.make("gym_hil/PandaPickCubeBase-v0", image_obs=True)
    obs, info = env.reset(seed=0)

    cams = list(obs["pixels"].keys())
    cam = cams[0]  # pick first camera

    plt.ion()
    fig, ax = plt.subplots()
    im = ax.imshow(obs["pixels"][cam])
    ax.set_title(f"Gym-HIL camera: {cam}")
    ax.axis("off")

    for _ in range(300):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        im.set_data(obs["pixels"][cam])
        fig.canvas.draw()
        fig.canvas.flush_events()

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    plt.ioff() 
    plt.show()

if __name__ == "__main__":
    main()
