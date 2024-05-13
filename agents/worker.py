import uuid
import time
# import asyncio
import numpy as np
import torch
import torch.distributed.rpc as rpc
# from torch.distributed.rpc import RRef
# from torch.futures import Future

from agents.worker_module.env_maker import EnvBase

from utils.utils import remote_method, remote_method_async, encode, Protocol, to_cpu_tensor


class WorkerRPC:
    def __init__(
        self, args, worker_name, obs_shape,
    ):
        self.args = args
        self.id = rpc.get_worker_info().id
        self.device = args.device  # cpu
        self.env = EnvBase(args)

        self.worker_name = worker_name

    def gene_rollout(self, learner_rref):
        from agents.learner import LearnerRPC as Learner
        print("Build Environment for {}".format(self.worker_name))

        self.num_epi = 0
        while True:
            obs = self.env.reset()
            # _id = uuid.uuid4().int
            _id = str(uuid.uuid4())  # 고유한 난수 생성

            # print(f"worker_name: {self.worker_name}, obs: {obs}")
            lstm_hx = (
                torch.zeros(self.args.hidden_size),
                torch.zeros(self.args.hidden_size),
            )  # (h_s, c_s) / (hidden,)
            self.epi_rew = 0

            is_fir = True  # first frame
            for _ in range(self.args.time_horizon):
                act, logits, log_prob, lstm_hx_next = to_cpu_tensor(*remote_method(Learner.act, learner_rref, self.id, obs, lstm_hx))
                next_obs, rew, done = self.env.step(act.item())

                self.epi_rew += rew

                step_data = {
                    "obs": obs,  # (c, h, w) or (D,)
                    "act": act.view(-1),  # (1,) / not one-hot, but action index
                    "rew": torch.from_numpy(
                        np.array([rew * self.args.reward_scale])
                    ),  # (1,)
                    "logits": logits,
                    "log_prob": log_prob.view(-1),  # (1,) / scalar
                    "is_fir": torch.FloatTensor([1.0 if is_fir else 0.0]),  # (1,),
                    "done": torch.FloatTensor([1.0 if done else 0.0]),  # (1,),
                    "hx": lstm_hx[0],  # (hidden,)
                    "cx": lstm_hx[1],  # (hidden,)
                    "id": _id,
                }
                remote_method(Learner.report_data, learner_rref, self.id, encode(Protocol.Rollout, step_data))
                
                is_fir = False
                obs = next_obs
                lstm_hx = lstm_hx_next

                time.sleep(0.05)

                if done:
                    break

            remote_method(Learner.report_data, learner_rref, self.id, encode(Protocol.Stat, self.epi_rew))
            
            self.epi_rew = 0
            self.num_epi += 1
