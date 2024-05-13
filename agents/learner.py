import asyncio
import copy
import numpy as np

from typing import List
from functools import partial

# import multiprocessing as mp
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.optim import Adam, RMSprop
from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote
# from torch.distributions import Categorical, Uniform

from utils.utils import (
    WORKER_NAME,
    make_gpu_batch,
    ExecutionTimer,
    Params,
    to_torch,
    call_method,
    decode,
)

from agents.worker import WorkerRPC
from agents.learner_storage import LearnerStorage

from .storage_module.shared_batch import SMInterFace
from . import (
    ppo_awrapper,
)

timer = ExecutionTimer(
    num_transition=Params.seq_len * Params.batch_size * 1
)  # Learner에서 데이터 처리량 (학습)


class LearnerRPC(SMInterFace):
    def __init__(
        self,
        args,
        model,
        shm_ref,
        stop_event,
        obs_shape,
    ):
        self.shm_ref = shm_ref
        super().__init__(shm_ref=shm_ref)
        
        # Setting Remote Reference
        self.learner_rref = RRef(self)
        self.wk_rrefs = []
        for i, wk_rank in enumerate(range(1, args.num_worker+1)): # worker-node + learner-node
            wk_name = "worker_" + str(i)
            wk_info = rpc.get_worker_info(WORKER_NAME.format(wk_rank))
            self.wk_rrefs.append(remote(wk_info, WorkerRPC, args=(args, wk_name, obs_shape,)))
            
        self.args = args
        self.stop_event = stop_event
        self.obs_shape = obs_shape
        
        self.device = self.args.device
        self.model = model.to(self.device)
        
        self.infer_model = copy.deepcopy(self.model)
        self.infer_model.eval()
        
        # self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        self.optimizer = RMSprop(self.model.parameters(), lr=self.args.lr, eps=1e-5)
        # self.CT = Categorical

        self.data_q = mp.Queue(maxsize=1024) # rollout, stat
        
        self.to_gpu = partial(make_gpu_batch, device=self.device)

        self.get_shared_memory_interface()

    def run(self):        
        grand_child_process = []
        
        storage = LearnerStorage(
            self.args,
            self.shm_ref,
            self.stop_event,
            self.obs_shape,
        )
        p1 = mp.Process(target=storage.entry_chain, args=(self.data_q, ), daemon=True)
        
        p2 = mp.Process(target=self.learner_entry_chain, daemon=True)
        grand_child_process.extend([p1])
        
        for gcp in grand_child_process:
            gcp.start()

        LearnerRPC.run_rref_worker(self.wk_rrefs, self.learner_rref)
 
        # for gcp in grand_child_process:
        #     gcp.join()

    def learner_entry_chain(self):
        asyncio.run(self.learning_chain_ppo())

    @staticmethod
    def run_rref_worker(wk_rrefs: List[RRef], learner_rref: RRef):
        futs = []
        for wk_rref in wk_rrefs:
            # make async RPC to kick off an episode on all observers
            futs.append(
                rpc_async(
                    wk_rref.owner(),
                    call_method,
                    args=(WorkerRPC.gene_rollout, wk_rref, learner_rref)
                )
            )

        for fut in futs:
            fut.wait()
            
    def act(self, wk_id, obs, lstm_hx):
        hx = lstm_hx[0].to(self.device)
        cx = lstm_hx[1].to(self.device)
        return self.infer_model.act(obs.to(self.device), (hx, cx))
    
    def report_data(self, wk_id, *data):
        self.data_q.put(decode(data))
    
    @staticmethod
    def copy_to_ndarray(src):
        dst = np.empty(src.shape, dtype=src.dtype)
        np.copyto(
            dst, src
        )  # 학습용 데이터를 새로 생성하고, 공유메모리의 데이터 오염을 막기 위함.
        return dst

    def sample_wrapper(func):
        def _wrap(self, sampling_method):
            sq = self.args.seq_len
            bn = self.args.batch_size
            sha = self.obs_shape
            # hs = self.args.hidden_size
            # ac = self.args.action_space
            # buf = self.args.buffer_size

            assert sampling_method == "on-policy"
            assert bn == int(
                self.sh_obs_batch.shape[0] / (sq * sha[0])
            )  # TODO: 좋은 코드는 아닌 듯..
            idx = slice(None)  # ":"와 동일
            num_buf = bn

            return func(self, idx, num_buf)

        return _wrap

    @sample_wrapper
    def sample_batch_from_sh_memory(self, idx, num_buf):
        sq = self.args.seq_len
        # bn = self.args.batch_size
        sha = self.obs_shape
        hs = self.args.hidden_size
        ac = self.args.action_space

        _sh_obs_batch = self.sh_obs_batch.reshape((num_buf, sq, *sha))[idx]
        _sh_act_batch = self.sh_act_batch.reshape((num_buf, sq, 1))[idx]
        _sh_rew_batch = self.sh_rew_batch.reshape((num_buf, sq, 1))[idx]
        _sh_logits_batch = self.sh_logits_batch.reshape((num_buf, sq, ac))[idx]
        _sh_log_prob_batch = self.sh_log_prob_batch.reshape((num_buf, sq, 1))[idx]
        _sh_is_fir_batch = self.sh_is_fir_batch.reshape((num_buf, sq, 1))[idx]
        _sh_hx_batch = self.sh_hx_batch.reshape((num_buf, sq, hs))[idx]
        _sh_cx_batch = self.sh_cx_batch.reshape((num_buf, sq, hs))[idx]

        # (batch, seq, feat)
        sh_obs_bat = LearnerRPC.copy_to_ndarray(_sh_obs_batch)
        sh_act_bat = LearnerRPC.copy_to_ndarray(_sh_act_batch)
        sh_rew_bat = LearnerRPC.copy_to_ndarray(_sh_rew_batch)
        sh_logits_bat = LearnerRPC.copy_to_ndarray(_sh_logits_batch)
        sh_log_prob_bat = LearnerRPC.copy_to_ndarray(_sh_log_prob_batch)
        sh_is_fir_bat = LearnerRPC.copy_to_ndarray(_sh_is_fir_batch)
        sh_hx_bat = LearnerRPC.copy_to_ndarray(_sh_hx_batch)
        sh_cx_bat = LearnerRPC.copy_to_ndarray(_sh_cx_batch)

        return (
            to_torch(sh_obs_bat),
            to_torch(sh_act_bat),
            to_torch(sh_rew_bat),
            to_torch(sh_logits_bat),
            to_torch(sh_log_prob_bat),
            to_torch(sh_is_fir_bat),
            to_torch(sh_hx_bat),
            to_torch(sh_cx_bat),
        )

    @ppo_awrapper(timer=timer)
    def learning_ppo(self): ...

    def is_sh_ready(self):
        bn = self.args.batch_size
        val = self.sh_data_num.value
        return True if val >= bn else False

    async def put_batch_to_batch_q(self):
        while not self.stop_event.is_set():
            if self.is_sh_ready():
                batch_args = self.sample_batch_from_sh_memory(
                    sampling_method="on-policy"
                )
                await self.batch_queue.put(batch_args)
                self.reset_data_num()  # 공유메모리 저장 인덱스 (batch_num) 초기화
                print("batch is ready !")

            await asyncio.sleep(0.001)

    async def learning_chain_ppo(self):
        self.batch_queue = asyncio.Queue(1024)
        tasks = [
            asyncio.create_task(self.learning_ppo()),
            asyncio.create_task(self.put_batch_to_batch_q()),
        ]
        await asyncio.gather(*tasks)