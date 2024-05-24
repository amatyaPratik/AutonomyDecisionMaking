import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import imageio
import keyboard

class PadmEnv(gym.Env):
    def __init__(self, grid_size=11):
        super().__init__()
        self.grid_size = grid_size
        self.agent_state = np.array([0, 0])
        self.goal_state = np.array([10, 10])
        self.obstacle_count = 10
        self.current_cumulative_reward = 0
        self.total_reward = 0
        self.goal_reached = False
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=self.grid_size, shape=(2,))
        self.fig, self.ax = plt.subplots()
        agent_path = os.path.join('.', 'pokemon', 'characters', 'ash.gif')
        self.agent_frames = imageio.mimread(agent_path)
        goal_path = os.path.join('.', 'pokemon', 'characters', 'pikachu.gif')
        self.goal_frames = imageio.mimread(goal_path)
        self.goal_reached_image_path = os.path.join('.', 'pokemon', 'assets', 'ash_finds_pikachu.gif')
        self.goal_reached_frames = imageio.mimread(self.goal_reached_image_path)
        self.walkable_path = os.path.join('.', 'pokemon', 'sprites', 'free-path.png')
        self.current_frame = 0
        self.obstacles = self.generate_obstacles()
        self.hell_states = self.generate_hell_states()
        self.danger_zone = False
        self.env_is_visible = 1
        plt.show(block=False)

    @staticmethod
    def degree_of_effect_zone(point1, point2):
        return max(abs(point1[0]-point2[0]), abs(point1[1]-point2[1]))

    @staticmethod
    def manhattan_distance(point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    def generate_obstacles(self):
        obstacles = []

        for i in range(self.obstacle_count):
            pos = np.array([np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)])
            while (np.array_equal(pos, self.agent_state) or
                   any(np.array_equal(pos, obs) for obs in obstacles) or
                   np.array_equal(pos, self.goal_state)):
                pos = np.array([np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)])
            obstacles.append(pos)
        return obstacles

    def generate_hell_states(self):
        hell_states_info = [
            ("haunter.gif", [-4,-2]),
            ("abra-2.gif", [-9,-5]),
            ("poisonous_flower.png", [-2]),
            ("poisonous_flower.png", [-2]),
            ("meowth.png", [-3]),
            ("mewtwo.gif", [-10])
        ]
        hell_states = []

        for filename, reward in hell_states_info:
            pos = np.array([np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)])
            while (np.array_equal(pos, self.agent_state) or 
                   any(np.array_equal(pos, obs) for obs in self.obstacles) or 
                   any(np.array_equal(pos, hs) for hs in hell_states)):
                pos = np.array([np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)])
            
            frames = imageio.mimread(os.path.join('.', 'pokemon', 'characters', filename))
            print(pos, filename)
            hell_states.append((pos, filename, frames, reward))

        return hell_states

    def reset(self):
        self.total_reward = 0
        self.agent_state = np.array([0, 0])
        self.goal_reached = False
        self.current_frame = 0
        return self.agent_state

    def step(self, action):
        self.env_is_visible += 1 if self.env_is_visible <= 0 else 0
        next_state = self.agent_state.copy()
        if action == "w" or action == "W":
            if self.agent_state[1] < self.grid_size - 1:  # up
                next_state[1] += 1
        elif action == "s" or action == "S":
            if self.agent_state[1] > 0:  # down
                next_state[1] -= 1
        elif action == "a" or action == "A":
            if self.agent_state[0] > 0:  # left
                next_state[0] -= 1
        elif action == "d" or action == "D":
            if self.agent_state[0] < self.grid_size - 1:  # right
                next_state[0] += 1

        if not any(np.array_equal(next_state, obs) for obs in self.obstacles):
            self.agent_state = next_state

        reward = 0
        done = np.array_equal(self.agent_state, self.goal_state)
        if done:
            reward = 10
            self.current_cumulative_reward = reward
            self.total_reward += reward
            self.goal_reached = True
        else:
            self.current_cumulative_reward = 0
            for pos, filename, frames, rew in self.hell_states:
                print(pos, filename)
                distance_from_hell_state = self.degree_of_effect_zone(self.agent_state, pos)
                print(pos , distance_from_hell_state)
                try:
                    reward = rew[distance_from_hell_state]
                    self.current_cumulative_reward += reward
                    if reward < 0:
                        if distance_from_hell_state == 0:
                            self.danger_zone = 0
                            if 'mewtwo' in filename:
                                self.reset()
                            if 'abra' in filename:
                                self.env_is_visible = -3
                        else:
                            self.danger_zone = distance_from_hell_state
                            
                except IndexError:
                    pass
                    # print(f"No value exists at index {distance_from_hell_state}")
                # break
            self.total_reward += self.current_cumulative_reward

        distance_to_goal = np.linalg.norm(self.goal_state - self.agent_state)
        info = {"Distance to Goal": distance_to_goal}

        return self.agent_state, self.current_cumulative_reward, done, info

    def render(self):
        self.ax.clear()
        self.ax.axis('off')

        if self.goal_reached:
            for frame in self.goal_reached_frames:
                self.ax.imshow(frame)
                plt.pause(0.1)
            plt.pause(1)
        else:
            walkable_image = mpimg.imread(self.walkable_path)
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    if self.env_is_visible > 0:
                        self.ax.imshow(walkable_image, extent=[x, x + 1, y, y + 1])

            for x in range(self.grid_size + 1):
                self.ax.plot([x, x], [0, self.grid_size], color='black')
            for y in range(self.grid_size + 1):
                self.ax.plot([0, self.grid_size], [y, y], color='black')

            agent_pos = self.agent_state + 0.5
            goal_pos = self.goal_state + 0.5

            if self.env_is_visible > 0:
                goal_frame = self.goal_frames[self.current_frame % len(self.goal_frames)]
                self.ax.imshow(goal_frame, extent=[goal_pos[0] - 0.5, goal_pos[0] + 0.5, goal_pos[1] - 0.5, goal_pos[1] + 0.5])

                # Draw vertical bars
                for i in np.linspace(goal_pos[0] - 0.5, goal_pos[0] + 0.5, num=4):
                    self.ax.plot([i, i], [goal_pos[1] - 0.5, goal_pos[1] + 0.5], color='black', linewidth=2)
                
                # Draw horizontal bars
                for i in np.linspace(goal_pos[1] - 0.5, goal_pos[1] + 0.5, num=4):
                    self.ax.plot([goal_pos[0] - 0.5, goal_pos[0] + 0.5], [i, i], color='black', linewidth=2)

            for obs in self.obstacles:
                obs_pos = obs + 0.5
                self.obstacle_path = os.path.join('.', 'pokemon', 'sprites', 'obstacle-2.png')
                obs_image = mpimg.imread(self.obstacle_path)
                if self.env_is_visible > 0:
                    self.ax.imshow(obs_image, extent=[obs_pos[0] - 0.5, obs_pos[0] + 0.5, obs_pos[1] - 0.5, obs_pos[1] + 0.5])

            for pos, filename, frames, _ in self.hell_states:
                hell_pos = pos + 0.5
                hell_frame = frames[self.current_frame % len(frames)]
                if self.env_is_visible > 0:
                    if np.array_equal(self.agent_state, pos):
                        # Clear the image at the specified cell
                        self.ax.imshow(np.zeros((1, 1)), extent=[pos[0], pos[0] + 1, pos[1], pos[1] + 1])
                        # Add a red rectangle to indicate danger
                        self.ax.add_patch(plt.Rectangle((pos[0], pos[1]), 1, 1, color='red'))
                    else:
                        self.ax.imshow(hell_frame, extent=[hell_pos[0] - 0.5, hell_pos[0] + 0.5, hell_pos[1] - 0.5, hell_pos[1] + 0.5])

            # Display the image of the agent
            if self.env_is_visible > 0:
                # if self.danger_zone:
                     # Clear the image at the specified cell
                    # self.ax.imshow(np.zeros((1, 1)), extent=[agent_pos[0], agent_pos[0] + 1, agent_pos[1], agent_pos[1] + 1])
                    # Add a red rectangle to indicate danger
                    # self.ax.add_patch(plt.Rectangle((agent_pos[0], agent_pos[1]), 1, 1, color='red'))
                    # self.danger_zone = False
                # else:
                    agent_frame = self.agent_frames[self.current_frame % len(self.agent_frames)]
                    self.ax.imshow(agent_frame, extent=[agent_pos[0] - 0.5, agent_pos[0] + 0.5, agent_pos[1] - 0.5, agent_pos[1] + 0.5])

            self.ax.set_xlim(0, self.grid_size)
            self.ax.set_ylim(0, self.grid_size)
            self.ax.set_aspect("equal")

            self.current_frame += 1

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
                if action in ["a", "s", "d", "w", "A", "S", "D", "W"]:
                    state, reward, done, info = env.step(action)
                    print(f"State:{state}, Reward:{reward}, Done:{done}, Info:{info}")
                    if done:
                        print("I reached the goal")
                        print(f"Total Reward = {env.total_reward}")
                        env.render()
                        print("Goal reached! Press Enter to close the window or BACKSPACE to restart.")
                        while True:
                            if keyboard.is_pressed("enter"):
                                env.close()
                                break
                            elif keyboard.is_pressed("backspace"):
                                env.reset()
                                break
                else:
                    print("Invalid action. Please enter a valid action.")
            except ValueError:
                print("Invalid input. Please enter a valid action.")
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
