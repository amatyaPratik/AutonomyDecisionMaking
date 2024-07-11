# NOTE: Code adapted from MinimalRL (URL: https://github.com/seungeunrho/minimalRL/blob/master/dqn.py)

# Imports:
# --------
import torch
import gymnasium as gym
from DQN_model import Qnet
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import ReplayBuffer, train
from padm_env import create_env
import numpy as np


# User definitions:
# -----------------
train_dqn = True
test_dqn = False
render = True

#! Define env attributes (environment specific)
no_actions = 4
no_states = 2

# Hyperparameters:
# ----------------
learning_rate = 0.0001
gamma = 0.99
buffer_limit = 50_000
batch_size = 64
num_episodes = 100
max_steps = 100
target_update_interval = 10

# Main:
# -----
if train_dqn:
    env = create_env()

    # Initialize the Q Net and the Q Target Net
    q_net = Qnet(no_actions=no_actions, no_states=no_states)
    q_target = Qnet(no_actions=no_actions, no_states=no_states)
    q_target.load_state_dict(q_net.state_dict())

    # Initialize the Replay Buffer
    memory = ReplayBuffer(buffer_limit=buffer_limit)

    print_interval = 10
    episode_reward = 0.0
    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    rewards = []

    for n_epi in range(num_episodes):
        epsilon = max(0.01, 1.0 - 0.005*(n_epi/1))  # Slower epsilon decay
        s, _, _ = env.reset()
        done = False

        for _ in range(max_steps):
            s = np.array(s).reshape(1, -1)
            a = q_net.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, _, r, done, _ = env.step(a)
            done_mask = 0.0 if done else 1.0

            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime
            episode_reward += r

            if done:
                break

        if memory.size() > 2000:
            train(q_net, q_target, memory, optimizer, batch_size, gamma)
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)

        if n_epi % target_update_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q_net.state_dict())
            print(f"n_episode :{n_epi}, Episode reward : {
                  episode_reward}, n_buffer : {memory.size()}, eps : {epsilon}")

        rewards.append(episode_reward)
        episode_reward = 0.0
        scheduler.step()

        # Define a stopping condition for the game:
        if len(rewards) >= 10 and all(r == max_steps for r in rewards[-10:]):
            break

    env.close()
    torch.save(q_net.state_dict(), "dqn.pth")

    plt.plot(rewards, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.legend()
    plt.savefig("training_curve.png")
    plt.show()

# Test:
if test_dqn:
    print("Testing the trained DQN: ")
    env = create_env()

    dqn = Qnet(no_actions=no_actions, no_states=no_states)
    dqn.load_state_dict(torch.load("dqn.pth"))

    for _ in range(10):
        s, _, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            action = dqn(torch.from_numpy(s).float())
            s_prime, _, r, done, _ = env.step(action.argmax().item())
            s = s_prime

            episode_reward += r

            if done:
                break
        print(f"Episode reward: {episode_reward}")

    env.close()
