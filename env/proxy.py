import gym
from gym import spaces

import numpy as np

from utils.utils import Params


env = gym.make(Params.env)

act_dim = (
    env.action_space.n 
    if hasattr(env.action_space, "n")
    else env.action_space.shape[0]
)
obs_dim = env.observation_space.shape[0]
env.close()

EnvSpace = spaces.Dict(
    obs=spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
    act=spaces.Discrete(act_dim),
    rew=spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32), # TODO: 하드코드
    info=spaces.Dict(
        hidden_size=spaces.Discrete(Params.hidden_size),
        seq_len=spaces.Discrete(Params.seq_len),
        batch_size=spaces.Discrete(Params.batch_size),
    )
)