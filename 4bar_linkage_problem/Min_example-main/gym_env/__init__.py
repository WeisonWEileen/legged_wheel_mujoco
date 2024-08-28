from gymnasium.envs.registration import register

register(
    id=f"Minimal_env",
    entry_point="gym_env.minimal_gym_env:MujocoGripEnv",
    kwargs={},
)