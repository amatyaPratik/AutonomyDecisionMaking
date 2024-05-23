import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os  # To handle relative paths
import imageio  # For loading and displaying GIFs

class PadmEnv(gym.Env):
    def __init__(self, grid_size=11):
        super().__init__()
        self.grid_size = grid_size
        self.agent_state = np.array([0, 0])
        self.goal_state = np.array([10, 10])
        self.obstacle_count = 10
        self.total_reward = 0
        self.goal_reached = False  # Add this flag
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=self.grid_size, shape=(2,))
        self.fig, self.ax = plt.subplots()
        agent_path = os.path.join('.', 'pokemon', 'characters', 'ash.gif')
        self.agent_image = mpimg.imread(agent_path)
        goal_path = os.path.join('.', 'pokemon', 'characters', 'pikachu.gif')
        self.goal_image = mpimg.imread(goal_path)
        self.goal_reached_image_path = os.path.join('.', 'pokemon', 'assets', 'ash_finds_pikachu.gif')
        self.walkable_path = os.path.join('.', 'pokemon', 'sprites', 'free-path.png')
        self.current_frame = 0
        self.obstacles = self.generate_obstacles()
        self.hell_states = self.generate_hell_states()
        self.env_is_visible = 1
        plt.show(block=False)

    def generate_obstacles(self):
        obstacles = []

        for i in range(self.obstacle_count):
            pos = np.array([np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)])
            while (np.array_equal(pos, self.agent_state) or
                    any(np.array_equal(pos, obs[0]) for obs in obstacles) or
                    np.array_equal(pos, self.goal_state)
                    ):
                pos = np.array([np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)])
            obstacles.append(pos)
        return obstacles

    def generate_hell_states(self):
        hell_states_info = [
            ("haunter.gif", -4),
            ("abra-2.gif", -9),
            ("jessie.png", -2),
            ("james.png", -2),
            ("meowth.png", -3),
            ("mewtwo.gif", -10)
        ]
        hell_states = []

        for filename, reward in hell_states_info:
            pos = np.array([np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)])
            while (np.array_equal(pos, self.agent_state) or 
                   any(np.array_equal(pos, obs) for obs in self.obstacles) or 
                   any(np.array_equal(pos, hs[0]) for hs in hell_states)):
                pos = np.array([np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)])
            
            hell_states.append((pos, filename, reward))

        return hell_states

    def reset(self):
        self.total_reward = 0
        self.agent_state = np.array([0, 0])
        self.goal_reached = False  # Reset the goal reached flag
        return self.agent_state

    def step(self, action):
        self.env_is_visible += 1 if self.env_is_visible<=0 else 0
        next_state = self.agent_state.copy()
        if action == "w" or action == "W" and self.agent_state[1] < self.grid_size - 1:  # up
            next_state[1] += 1
        elif action == "s" or action == "S" and self.agent_state[1] > 0:  # down
            next_state[1] -= 1
        elif action == "a" or action == "A" and self.agent_state[0] > 0:  # left
            next_state[0] -= 1
        elif action == "d" or action == "D" and self.agent_state[0] < self.grid_size - 1:  # right
            next_state[0] += 1

        if not any(np.array_equal(next_state, obs) for obs in self.obstacles):
            self.agent_state = next_state

        reward = 0
        done = np.array_equal(self.agent_state, self.goal_state)
        if done:
            reward = 10
            self.total_reward += reward
            self.goal_reached = True  # Set the goal reached flag
        else:
            for pos, name, rew in self.hell_states:
                if np.array_equal(self.agent_state, pos):
                    reward = rew
                    self.total_reward += reward
                    if 'mewtwo' in name:
                        self.reset()
                    elif 'abra' in name:
                        self.env_is_visible = -3
                    break

        distance_to_goal = np.linalg.norm(self.goal_state - self.agent_state)
        info = {"Distance to Goal": distance_to_goal}

        return self.agent_state, reward, done, info

    def render(self):
        self.ax.clear()
        self.ax.axis('off')

        if self.goal_reached:
            goal_reached_image = imageio.mimread(self.goal_reached_image_path)
            plt.imshow(goal_reached_image[0])  # Display the first frame of the GIF
            for frame in goal_reached_image:
                plt.imshow(frame)
                plt.pause(0.1)
        else:
            walkable_image = mpimg.imread(self.walkable_path)
            if self.env_is_visible > 0:
                self.ax.imshow(walkable_image, extent=[0, self.grid_size, 0, self.grid_size])

            for x in range(self.grid_size + 1):
                self.ax.plot([x, x], [0, self.grid_size], color='black')
            for y in range(self.grid_size + 1):
                self.ax.plot([0, self.grid_size], [y, y], color='black')

            agent_pos = self.agent_state + 0.5
            goal_pos = self.goal_state + 0.5

            if self.env_is_visible > 0:
                self.ax.imshow(self.goal_image, extent=[goal_pos[0] - 0.5, goal_pos[0] + 0.5, goal_pos[1] - 0.5, goal_pos[1] + 0.5])

            for obs in self.obstacles:
                obs_pos = obs + 0.5
                self.obstacle_path = os.path.join('.', 'pokemon', 'sprites', 'obstacle.png')
                obs_image = mpimg.imread(self.obstacle_path)
                if self.env_is_visible > 0:
                    self.ax.imshow(obs_image, extent=[obs_pos[0] - 0.5, obs_pos[0] + 0.5, obs_pos[1] - 0.5, obs_pos[1] + 0.5])

            for pos, filename, _ in self.hell_states:
                hell_pos = pos + 0.5
                hell_image = mpimg.imread(os.path.join('.', 'pokemon', 'characters', filename))
                if self.env_is_visible > 0:
                    self.ax.imshow(hell_image, extent=[hell_pos[0] - 0.5, hell_pos[0] + 0.5, hell_pos[1] - 0.5, hell_pos[1] + 0.5])

            if self.env_is_visible > 0:
                self.ax.imshow(self.agent_image, extent=[agent_pos[0] - 0.5, agent_pos[0] + 0.5, agent_pos[1] - 0.5, agent_pos[1] + 0.5])

            self.ax.set_xlim(0, self.grid_size)
            self.ax.set_ylim(0, self.grid_size)
            self.ax.set_aspect("equal")

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
            action = input("Enter action (w=up, s=down, a=left, d=right): ")
            try:
                if action in ["a","s","d","w","A","S","D","W"]:
                    state, reward, done, info = env.step(action)
                    print(f"State:{state}, Reward:{reward}, Done:{done}, Info:{info}")
                    if done:
                        print("I reached the goal")
                        print(f"Total Reward = {env.total_reward}")
                        env.render()  # Render the final GIF display
                        input("Press Enter to close the window.")  # Wait for user input to close the window
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
