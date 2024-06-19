# Imports:
# --------
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Function 1: Train Q-learning agent
# -----------


def train_q_learning(env,
                     renderChoice,
                     no_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     q_table_save_path="q_table.npy"):

    # Initialize the Q-table:
    # -----------------------
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

    # Q-learning algorithm:
    # ---------------------
    #! Step 1: Run the algorithm for fixed number of episodes
    #! -------
    for episode in range(no_episodes):
        state, _, _ = env.reset()

        state = tuple(state)
        total_reward = 0

        #! Step 2: Take actions in the environment until "Done" flag is triggered
        #! -------
        while True:
            #! Step 3: Define your Exploration vs. Exploitation
            #! -------
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            # return self.agent_state, observation, self.current_cumulative_reward, done, info

            next_state, _, reward, done, _ = env.step(action)

            if renderChoice:
                env.render()

            next_state = tuple(next_state)
            print(f"next_state: {next_state}")
            total_reward += reward
            print(f"total_reward: {total_reward}")

            #! Step 4: Update the Q-values using the Q-value update rule
            #! -------
            q_table[state][action] = q_table[state][action] + alpha * \
                (reward + gamma *
                 np.max(q_table[next_state]) - q_table[state][action])

            state = next_state

            #! Step 5: Stop the episode if the agent reaches Goal or Hell-states
            #! -------
            if done:
                break

        #! Step 6: Perform epsilon decay
        #! -------
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    #! Step 7: Close the environment window
    #! -------
    env.close()
    print("Training finished.\n")

    #! Step 8: Save the trained Q-table
    #! -------
    np.save(q_table_save_path, q_table)
    print("Saved the Q-table.")

# Function 2: Visualize the Q-table
# -----------


def visualize_q_table(hell_state_coordinates=[(0, 1), (5, 0), (6, 4), (6, 5), (1, 5), (4, 5), (9, 5), (4, 10), (8, 8), (10, 8), (9, 5)],
                      goal_coordinates=(10, 10),
                      actions=["Up", "Down", "Right", "Left"],
                      q_values_path="q_table.npy"):

    # Load the Q-table:
    # -----------------
    try:
        q_table = np.load(q_values_path)

        # Create subplots for each action:
        # --------------------------------
        _, axes = plt.subplots(1, 4, figsize=(20, 5))

        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()

            # Mask the goal state's Q-value for visualization:
            # ------------------------------------------------
            mask = np.zeros_like(heatmap_data, dtype=bool)
            mask[goal_coordinates] = True
            mask[hell_state_coordinates[0]] = True
            mask[hell_state_coordinates[1]] = True

            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis",
                        ax=ax, cbar=False, mask=mask, annot_kws={"size": 9})

            # Denote Goal and Hell states:
            # ----------------------------
            ax.text(goal_coordinates[1] + 0.5, goal_coordinates[0] + 0.5, 'G', color='green',
                    ha='center', va='center', weight='bold', fontsize=14)
            ax.text(hell_state_coordinates[0][1] + 0.5, hell_state_coordinates[0][0] + 0.5, 'H', color='red',
                    ha='center', va='center', weight='bold', fontsize=14)
            ax.text(hell_state_coordinates[1][1] + 0.5, hell_state_coordinates[1][0] + 0.5, 'H', color='red',
                    ha='center', va='center', weight='bold', fontsize=14)

            ax.set_title(f'Action: {action}')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")
