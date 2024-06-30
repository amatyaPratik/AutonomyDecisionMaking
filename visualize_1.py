
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

hell_state_coordinates = [(0, 1), (5, 0), (6, 4), (7, 4),
                          (1, 5), (4, 5), (9, 5), (4, 10), (8, 8), (10, 8)]
goal_coordinates = (10, 10)
actions = ["Up", "Down", "Right", "Left"]
q_values_path = "q_table.npy"


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
    for i in hell_state_coordinates:
        mask[(i[1], i[0])] = True

    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis",
                ax=ax, cbar=False, mask=mask, annot_kws={"size": 9})

    # Denote Goal and Hell states:
    # ----------------------------
    ax.text(goal_coordinates[1] + 0.5, goal_coordinates[0] + 0.5, 'G', color='green',
            ha='center', va='center', weight='bold', fontsize=14)

    for i in hell_state_coordinates:
        ax.text(i[0] + 0.5, i[1] + 0.5, 'H', color='red',
                ha='center', va='center', weight='bold', fontsize=14)

    ax.set_title(f'Action: {action}')
    ax.invert_yaxis()  # Invert the y-axis to increase from bottom to top

plt.tight_layout()
plt.show()
