import asyncio
import numpy as np
import torch.multiprocessing as mp
# import multiprocessing as mp
from collections import deque

from .storage_module.shared_batch import SMInterFace

from buffers.rollout_assembler import RolloutAssembler

from utils.utils import (
    Protocol,
    mul,
    flatten,
    counted,

)


class LearnerStorage(SMInterFace):
    def __init__(
        self,
        args,
        shm_ref,
        stop_event,
        obs_shape,
    ):
        super().__init__(shm_ref=shm_ref)

        self.args = args
        self.stop_event = stop_event
        self.obs_shape = obs_shape

        self.stat_publish_cycle = 50
        self.stat_q = deque(maxlen=self.stat_publish_cycle)

        self.get_shared_memory_interface()

    @staticmethod
    def entry_chain(self, data_queue: mp.Queue):
        asyncio.run(self.shared_memory_chain(data_queue))

    async def shared_memory_chain(self, data_queue: mp.Queue):
        self.rollout_assembler = RolloutAssembler(self.args, asyncio.Queue(1024))

        tasks = [
            asyncio.create_task(self.retrieve_rollout_from_worker(data_queue)),
            asyncio.create_task(self.build_as_batch()),
        ]
        await asyncio.gather(*tasks)

    async def retrieve_rollout_from_worker(self, data_queue: mp.Queue):
        while not self.stop_event.is_set():
            if not data_queue.empty():
                protocol, data = data_queue.get()  # FIFO
                
                if protocol is Protocol.Rollout:
                    await self.rollout_assembler.push(data)
                    
                elif protocol is Protocol.Stat:
                    self.stat_q.append(data)
                    if stat_pub_num >= self.stat_publish_cycle and len(self.stat_q) > 0:
    
                        self.log_stat(
                            {"log_len": len(self.stat_q), "mean_stat": np.mean([self.stat_q])}
                        )
                        stat_pub_num = 0
                    stat_pub_num += 1
                    
                else:
                    assert False, f"Wrong protocol: {protocol}"

            await asyncio.sleep(0.001)

    async def build_as_batch(self):
        while not self.stop_event.is_set():
            # with timer.timer("learner-storage-throughput", check_throughput=True):
            data = await self.rollout_assembler.pop()
            self.make_batch(data)
            print("rollout is poped !")

            await asyncio.sleep(0.001)

    @counted
    def log_stat(self, data):
        _len = data["log_len"]
        _epi_rew = data["mean_stat"]

        tag = f"worker/{_len}-game-mean-stat-of-epi-rew"
        x = self.log_stat.calls * _len  # global game counts
        y = _epi_rew

        print(f"tag: {tag}, y: {y}, x: {x}")


    def make_batch(self, rollout):
        sq = self.args.seq_len
        # bn = self.args.batch_size

        num = self.sh_data_num.value

        sha = mul(self.obs_shape)
        ac = self.args.action_space
        hs = self.args.hidden_size

        # buf = self.args.buffer_size
        mem_size = int(
            self.sh_obs_batch.shape[0] / (sq * sha)
        )  # TODO: 좋은 코드는 아닌 듯..
        # assert buf == mem_size

        if num < mem_size:
            obs = rollout["obs"]
            act = rollout["act"]
            rew = rollout["rew"]
            logits = rollout["logits"]
            log_prob = rollout["log_prob"]
            is_fir = rollout["is_fir"]
            hx = rollout["hx"]
            cx = rollout["cx"]

            # 공유메모리에 학습 데이터 적재
            self.sh_obs_batch[sq * num * sha : sq * (num + 1) * sha] = flatten(obs)
            self.sh_act_batch[sq * num : sq * (num + 1)] = flatten(act)
            self.sh_rew_batch[sq * num : sq * (num + 1)] = flatten(rew)
            self.sh_logits_batch[sq * num * ac : sq * (num + 1) * ac] = flatten(logits)
            self.sh_log_prob_batch[sq * num : sq * (num + 1)] = flatten(log_prob)
            self.sh_is_fir_batch[sq * num : sq * (num + 1)] = flatten(is_fir)
            self.sh_hx_batch[sq * num * hs : sq * (num + 1) * hs] = flatten(hx)
            self.sh_cx_batch[sq * num * hs : sq * (num + 1) * hs] = flatten(cx)

            self.sh_data_num.value += 1