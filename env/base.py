import gym

from utils.utils import to_torch


class EnvBase:
    def __init__(self, args):
        self.args = args
        self._env = gym.make(args.env)

    def reset(self):
        obs, _ = self._env.reset()
        return to_torch(obs)

    def step(self, act):
        obs, rew, done, _, _ = self._env.step(act)
        return to_torch(obs), rew, done
