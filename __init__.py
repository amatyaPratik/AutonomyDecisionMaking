from gymnasium.envs.registration import register

register(
    id="GridWorld-v0",
    entry_point="./GridWorldEnv.py",
    max_episode_steps=300,
)