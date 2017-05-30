from gym.envs.registration import register

from .envs.nerfbot_env import OneStaticCircleTarget

environments = [ ['OneStaticCircleTarget', 'v0'] ]

for environment in environments:
    register(
        id='{}-{}'.format(environment[0], environment[1]),
        entry_point='gym_nerfbot.envs:{}'.format(environment[0]),
        timestep_limit=1000,
        reward_threshold=1000.0,
        nondeterministic=False,
    )
