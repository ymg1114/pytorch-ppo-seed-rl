import asyncio
import numpy as np
from collections import deque

from .storage_module.batch_memory import NdaMemInterFace

from buffers.rollout_assembler import RolloutAssembler

from utils.utils import (
    Protocol,
    mul,
    flatten,
)


class LearnerStorage(NdaMemInterFace):
    def __init__(
        self,
        args,
        nda_ref,
        stop_event,
        obs_shape,
    ):
        super().__init__(nda_ref=nda_ref)
        self.get_nda_memory_interface()
        
        self.args = args
        self.stop_event = stop_event
        self.obs_shape = obs_shape

    @staticmethod
    def entry_chain(self, data_queue: deque):
        asyncio.run(self.memory_chain(data_queue))

    async def memory_chain(self, data_queue: deque):
        self.rollout_assembler = RolloutAssembler(self.args, asyncio.Queue(1024))

        tasks = [
            asyncio.create_task(self.retrieve_rollout_from_worker(data_queue)),
            asyncio.create_task(self.build_as_batch()),
        ]
        await asyncio.gather(*tasks)

    async def retrieve_rollout_from_worker(self, data_queue: deque):
        while not self.stop_event.is_set():
            if len(data_queue) > 0:
                protocol, data = data_queue.popleft()  # FIFO
                
                assert protocol is Protocol.Rollout
                await self.rollout_assembler.push(data) 

            await asyncio.sleep(0.001)

    async def build_as_batch(self):
        while not self.stop_event.is_set():
            # with timer.timer("learner-storage-throughput", check_throughput=True):
            data = await self.rollout_assembler.pop()
            self.make_batch(data)
            print("rollout is poped !")

            await asyncio.sleep(0.001)

    def make_batch(self, rollout):
        sq = self.args.seq_len
        # bn = self.args.batch_size

        num = int(self.nda_data_num.item())

        sha = mul(self.obs_shape)
        ac = self.args.action_space
        hs = self.args.hidden_size

        # buf = self.args.buffer_size
        mem_size = int(
            self.nda_obs_batch.shape[0] / (sq * sha)
        )  # TODO: 좋은 코드는 아닌 듯..
        # assert buf == mem_size

        if num < mem_size:
            obs = rollout["obs"].cpu()
            act = rollout["act"].cpu()
            rew = rollout["rew"].cpu()
            logits = rollout["logits"].cpu()
            log_prob = rollout["log_prob"].cpu()
            is_fir = rollout["is_fir"].cpu()
            hx = rollout["hx"].cpu()
            cx = rollout["cx"].cpu()

            # 메모리에 학습 데이터 적재
            self.nda_obs_batch[sq * num * sha : sq * (num + 1) * sha] = flatten(obs)
            self.nda_act_batch[sq * num : sq * (num + 1)] = flatten(act)
            self.nda_rew_batch[sq * num : sq * (num + 1)] = flatten(rew)
            self.nda_logits_batch[sq * num * ac : sq * (num + 1) * ac] = flatten(logits)
            self.nda_log_prob_batch[sq * num : sq * (num + 1)] = flatten(log_prob)
            self.nda_is_fir_batch[sq * num : sq * (num + 1)] = flatten(is_fir)
            self.nda_hx_batch[sq * num * hs : sq * (num + 1) * hs] = flatten(hx)
            self.nda_cx_batch[sq * num * hs : sq * (num + 1) * hs] = flatten(cx)

            self.nda_data_num[:] += 1