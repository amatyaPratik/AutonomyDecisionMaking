import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

obstacle_state_coordinates = [(0, 2), (1, 7), (1, 8), (2, 1), (2, 3), (2, 8), (2, 9), (3, 1), (3, 3), (3, 6), (3, 9), (4, 6), (5, 2), (5, 4), (
    5, 5), (5, 6), (5, 9), (6, 2), (6, 8), (6, 9), (7, 0), (7, 5), (7, 6), (8, 0), (8, 1), (8, 5), (8, 6), (8, 9), (10, 3), (10, 4), (10, 5), (10, 6)]

# Load the Q-table
q_values_path = "q_table.npy"
actions = ["Up", "Right", "Down", "Left"]
q_table = np.load(q_values_path)

# Initialize the optimal action data with the shape of the state space (first two dimensions of q_table)
optimal_action_data = np.zeros(q_table.shape[:2], dtype=int)

# Initialize the optimal heatmap data with the first action's values
optimal_heatmap_data = q_table[:, :, 0].copy()

# Iterate through the remaining actions and update the optimal heatmap and action data
for idx in range(1, len(actions)):
    heatmap_data = q_table[:, :, idx]
    # Create a mask where current action's Q-values are greater
    mask = heatmap_data > optimal_heatmap_data
    # Update the maximum Q-values
    optimal_heatmap_data[mask] = heatmap_data[mask]
    optimal_action_data[mask] = idx  # Update the optimal actions

obs_mask = np.zeros_like(heatmap_data, dtype=bool)
for obs in obstacle_state_coordinates:
    obs_mask[(obs[1], obs[0])] = True

# Create a 2D array of action labels corresponding to the optimal action indices
action_labels = np.array(actions)[optimal_action_data]
action_labels = action_labels.transpose()

# Plot the optimal actions using seaborn heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(optimal_action_data, annot=action_labels,
                 fmt='', cmap='coolwarm', cbar=False, mask=obs_mask)
ax.set_title('Optimal Actions for Each State')
ax.set_xlabel('State X')
ax.set_ylabel('State Y')
ax.invert_yaxis()
plt.show()
