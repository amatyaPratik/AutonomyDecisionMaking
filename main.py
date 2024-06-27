# Imports:
# --------
from GridWorldEnv import create_env
from Q_learning import train_q_learning, visualize_q_table

# User definitions:
# -----------------
train = True
visualize_results = False

learning_rate = 0.01  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
no_episodes = 50_000  # Number of episodes

# Execute:
# --------
if train:
    # Create an instance of the environment:
    # --------------------------------------
    env = create_env()

    # Train a Q-learning agent:
    # -------------------------
    train_q_learning(env=env,
                     renderChoice=False,
                     no_episodes=no_episodes,
                     epsilon=epsilon,
                     epsilon_min=epsilon_min,
                     epsilon_decay=epsilon_decay,
                     alpha=learning_rate,
                     gamma=gamma)

if visualize_results:
    # Visualize the Q-table:
    # ----------------------
    visualize_q_table()
