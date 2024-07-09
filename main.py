# Imports:
# --------
import numpy as np
from DQNAgent import DQNAgent
from GridWorldEnv import create_env
from keras._tf_keras.keras import Sequential
from keras._tf_keras.keras.layers import Dense
from keras._tf_keras.keras.layers import Flatten
import matplotlib.pyplot as plt

train_not_visualize = True
no_episodes = 10

if train_not_visualize:
    # Create an instance of the environment:
    # --------------------------------------
    env = create_env()  # Replace with your environment class
    state_size = 2  # Since the state is represented by (x, y)
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    episode_rewards = []

    for e in range(no_episodes):  # Number of episodes
        state, _, _ = env.reset()
        print(np.shape(state))
        state = np.array(state).reshape(1, -1)
        total_reward = 0

        done = False
        for time in range(500):
            action = agent.act(state)
            next_state, _, reward, done, _ = env.step(action)
            next_state = np.array(state).reshape(1, -1)
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {
                      e+1}/{no_episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        episode_rewards.append(total_reward)
        # Save model after every episode
        agent.save(f"dqn_model_{e}.weights.h5")
    # Print average reward per episode
    average_reward = np.mean(episode_rewards)
    print(f"Average Reward per Episode: {average_reward}")
    # Plotting the training curve
    plt.plot(range(1, no_episodes + 1), episode_rewards, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Curve')
    plt.grid(True)
    plt.show()

else:  # if not train_visualize:
    # Define the model architecture
    model = Sequential([
        Flatten(input_shape=(2,)),  # Input layer flattens the 11x11 grid
        Dense(24, activation='relu'),
        Dense(24, activation='relu'),
        Dense(4, activation='linear')   # Output layer with 4 actions
    ])

    state = np.array([0, 0])

    # Load weights into model
    model.load_weights('./dqn_model_0.weights.h5')

    def choose_action(state):
        print(f"state: {state}")
        q_values = model.predict(np.array([state]))[0]
        return np.argmax(q_values)  # Choose action with highest Q-value

    # Print model summary
    print(model.summary())

    env = create_env()
    done = False
    while not done:
        action = choose_action(state)
        agent_state, observation, current_cumulative_reward, done, info = env.step(
            action)
        state = agent_state
