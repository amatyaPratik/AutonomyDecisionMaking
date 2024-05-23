import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

class PadmEnv(gym.Env):
    def __init__(self, grid_size=11):
        super().__init__()
        self.grid_size = grid_size
        self.agent_state = np.array([1, 1])
        self.goal_state = np.array([10, 10])
        self.obstacles = [np.array([2, 3]), np.array([5, 7]), np.array([3, 9])]
        self.hell_state = np.array([7, 7])
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=self.grid_size, shape=(2,))
        self.fig, self.ax = plt.subplots()
        self.agent_frames = self.load_gif_frames(r'C:\Users\prati\UNI\autonomy_n_decision_making\my-works\week-3-assignment-1\pokemon\characters\ash.gif')
        self.current_frame = 0
        plt.show(block=False)

    def load_gif_frames(self, gif_path):
        gif = Image.open(gif_path)
        frames = []
        try:
            while True:
                frame = gif.copy().convert('RGBA')
                frames.append(np.array(frame))
                gif.seek(len(frames))
        except EOFError:
            pass
        return frames

    def reset(self):
        self.agent_state = np.array([1, 1])
        return self.agent_state

    def step(self, action):
        next_state = self.agent_state.copy()
        if action == 0 and self.agent_state[1] < self.grid_size - 1:  # up
            next_state[1] += 1
        elif action == 1 and self.agent_state[1] > 0:  # down
            next_state[1] -= 1
        elif action == 2 and self.agent_state[0] > 0:  # left
            next_state[0] -= 1
        elif action == 3 and self.agent_state[0] < self.grid_size - 1:  # right
            next_state[0] += 1

        if not any(np.array_equal(next_state, obs) for obs in self.obstacles):
            self.agent_state = next_state

        reward = 0
        done = np.array_equal(self.agent_state, self.goal_state)
        if done:
            reward = 10
        elif np.array_equal(self.agent_state, self.hell_state):
            reward = -5

        # Now we use Euclidean Distance
        distance_to_goal = np.linalg.norm(self.goal_state - self.agent_state)
        info = {"Distance to Goal": distance_to_goal}

        return self.agent_state, reward, done, info

    def render(self):
        self.ax.clear()

        # Draw grid lines
        for x in range(self.grid_size + 1):
            self.ax.plot([x, x], [0, self.grid_size], color='black')
        for y in range(self.grid_size + 1):
            self.ax.plot([0, self.grid_size], [y, y], color='black')

        # Calculate positions to center the agent and goal in the grid cells
        agent_pos = self.agent_state + 0.5
        goal_pos = self.goal_state + 0.5

        # Draw goal, obstacles, and hell state
        self.ax.plot(goal_pos[0], goal_pos[1], "g+")

        for obs in self.obstacles:
            obs_pos = obs + 0.5
            self.ax.plot(obs_pos[0], obs_pos[1], "kx")  # 'kx' for black 'x'

        hell_pos = self.hell_state + 0.5
        self.ax.text(hell_pos[0], hell_pos[1], "-", color="red", fontsize=20, ha="center", va="center")

        # Display the agent image (animated GIF frame)
        agent_img = self.agent_frames[self.current_frame]
        self.ax.imshow(agent_img, extent=[agent_pos[0]-0.5, agent_pos[0]+0.5, agent_pos[1]-0.5, agent_pos[1]+0.5])

        # Update to the next frame
        self.current_frame = (self.current_frame + 1) % len(self.agent_frames)

        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_aspect("equal")

        # Hide the axes
        self.ax.axis('off')

        plt.pause(0.1)

    def close(self):
        plt.close()

if __name__ == "__main__":
    env = PadmEnv()
    state = env.reset()
    done = False
    try:
        while not done:
            env.render()
            action = input("Enter action (0=up, 1=down, 2=left, 3=right): ")
            try:
                action = int(action)
                if action in [0, 1, 2, 3]:
                    state, reward, done, info = env.step(action)
                    print(f"State:{state}, Reward:{reward}, Done:{done}, Info:{info}")
                    if done:
                        print("I reached the goal")
                    elif reward == -5:
                        print("I entered the hell state")
                else:
                    print("Invalid action. Please enter a number between 0 and 3.")
            except ValueError:
                print("Invalid input. Please enter an integer.")
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
