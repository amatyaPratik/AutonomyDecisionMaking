import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random


class DQNAgent:
    def __init__(self):
        self.state_size = 2
        self.action_size = 4
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, [1, self.state_size])
            next_state = np.reshape(next_state, [1, self.state_size])
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t[0])
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# Training the DQN Agent
if __name__ == "__main__":
    agent = DQNAgent()
    num_episodes = 1000
    batch_size = 32
    state_space = (11, 11)

    for e in range(num_episodes):
        state = np.array([0, 0])
        for time in range(500):
            action = agent.act(state)
            next_state = state.copy()
            if action == 0 and state[1] > 0:  # up
                next_state[1] -= 1
            elif action == 1 and state[1] < state_space[1] - 1:  # down
                next_state[1] += 1
            elif action == 2 and state[0] > 0:  # left
                next_state[0] -= 1
            elif action == 3 and state[0] < state_space[0] - 1:  # right
                next_state[0] += 1

            reward = -1  # Define your own reward structure
            done = np.array_equal(next_state, [10, 10])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f"Episode: {
                      e}/{num_episodes}, Time: {time}, Epsilon: {agent.epsilon:.2}")
                break

            agent.replay(batch_size)
        if e % 10 == 0:
            agent.update_target_model()
