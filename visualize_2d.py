import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

hell_state_coordinates = [(0, 1), (5, 0), (6, 4), (7, 4),
                          (1, 5), (4, 5), (9, 5), (4, 10), (8, 8), (10, 8)]
obstacle_state_coordinates = [(0, 2), (1, 7), (1, 8), (2, 1), (2, 3), (2, 8), (2, 9), (3, 1), (3, 3), (3, 6), (3, 9), (4, 6), (5, 2), (5, 4), (
    5, 5), (5, 6), (5, 9), (6, 2), (6, 8), (6, 9), (7, 0), (7, 5), (7, 6), (8, 0), (8, 1), (8, 5), (8, 6), (8, 9), (10, 3), (10, 4), (10, 5), (10, 6)]
goal_coordinates = (10, 10)
actions = ["Up", "Right", "Down", "Left"]
q_values_path = "q_table.npy"

q_table = np.load(q_values_path)

# Create subplots for each action in a 2x2 grid:
# ----------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for idx, action in enumerate(actions):
    ax = axes[idx // 2, idx % 2]
    heatmap_data = q_table[:, :, idx].copy()
    heatmap_data = heatmap_data.transpose()

    # Mask the goal state's Q-value for visualization:
    # ------------------------------------------------
    mask = np.zeros_like(heatmap_data, dtype=bool)
    mask[goal_coordinates] = True
    for coord in hell_state_coordinates:
        mask[(coord[1], coord[0])] = True

    for obs in obstacle_state_coordinates:
        mask[(obs[1], obs[0])] = True

    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis",
                ax=ax, cbar=False, mask=mask, annot_kws={"size": 9})

    # Denote Goal and Hell states:
    # ----------------------------
    ax.text(goal_coordinates[0] + 0.5, goal_coordinates[1] + 0.5, 'G', color='green',
            ha='center', va='center', weight='bold', fontsize=14)

    for coord in hell_state_coordinates:
        ax.text(coord[0] + 0.5, coord[1] + 0.5, 'H', color='red',
                ha='center', va='center', weight='bold', fontsize=14)

    for obs in obstacle_state_coordinates:
        ax.text(obs[0] + 0.5, obs[1] + 0.5, 'X', color='black',
                ha='center', va='center', weight='bold', fontsize=14)

    ax.set_title(f'Action: {action}')
    ax.invert_yaxis()  # Invert the y-axis to increase from bottom to top

plt.tight_layout()
plt.show()
