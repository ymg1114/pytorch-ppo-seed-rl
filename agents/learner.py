import asyncio
import copy
import threading as th

from collections import deque, defaultdict
from typing import List
from functools import partial

import torch.jit as jit
import torch.distributed.rpc as rpc
from torch.futures import Future
from torch.optim import Adam, RMSprop
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote

from networks.models import MlpLSTM

from utils.utils import (
    WORKER_NAME,
    set_model_weight,
    Protocol,
    make_gpu_batch,
    ExecutionTimer,
    Params,
    to_torch,
    call_method,
    decode,
)
from env.proxy import EnvSpace

from agents.worker import WorkerRPC
from agents.learner_storage import LearnerStorage

from .storage_module.batch_memory import NdaMemInterFace
from . import (
    ppo_awrapper,
)

timer = ExecutionTimer(
    num_transition=Params.seq_len * Params.batch_size * 1
)  # Learner에서 데이터 처리량 (학습)


class LearnerRPC(NdaMemInterFace):
    def __init__(
        self,
        args,
        nda_ref,
        obs_shape,
    ):
        super().__init__(nda_ref=nda_ref)
        self.get_nda_memory_interface()
        
        self.args = args
        self.id = rpc.get_worker_info().id
        self.obs_shape = obs_shape
        
        self.data_q = deque(maxlen=1024) # rollout, stat queue
        self.stop_event = th.Event() # condition
        
        # Reset
        self.step_data = defaultdict(dict)
        
        self.device = self.args.device
        
        # self.model = jit.script(MlpLSTM(args=Params, env_space=EnvSpace)).to(self.device)
        self.model = MlpLSTM(args=Params, env_space=EnvSpace).to(self.device)
        learner_model_state_dict = set_model_weight(self.args)
        if learner_model_state_dict is not None:
            self.model.load_state_dict(learner_model_state_dict)
            
        self.infer_model = copy.deepcopy(self.model)
        self.infer_model.eval()
        
        # Setting Remote Reference
        self.learner_rref = RRef(self)
        self.wk_rrefs = []
        for i, wk_rank in enumerate(range(1, args.num_worker+1)): # worker-node + learner-node
            wk_name = "worker_" + str(i)
            wk_info = rpc.get_worker_info(WORKER_NAME.format(wk_rank))
            self.wk_rrefs.append(remote(wk_info, WorkerRPC, args=(args, wk_name,)))
        
        self.to_gpu = partial(make_gpu_batch, device=self.device)
        self.optimizer = RMSprop(self.model.parameters(), lr=self.args.lr, eps=1e-5)

    def run(self):        
        grand_child_thread = []
        
        storage = LearnerStorage(
            self.args,
            self.nda_ref,
            self.stop_event,
            self.obs_shape,
        )
        t1 = th.Thread(target=storage.entry_chain, args=(storage, self.data_q, ), daemon=True)
        t2 = th.Thread(target=self.entry_chain, daemon=True)
        grand_child_thread.extend([t1, t2])
        
        for gcp in grand_child_thread:
            gcp.start()

        LearnerRPC.run_rref_worker(self.wk_rrefs, self.learner_rref)
 
        # for gcp in grand_child_thread:
        #     gcp.join()

    def entry_chain(self):
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
                    args=(WorkerRPC.gene_rollout, wk_rref, learner_rref),
                    timeout=0,
                )
            )
        for fut in futs:
            fut.wait()

    def proxy_step_data(self, wk_id, obs, lstm_hx, results):
        act, logits, log_prob, _ = results
        
        self.step_data[wk_id] = {
            "obs": obs,  # (c, h, w) or (D,)
            "act": act.view(-1),  # (1,) / not one-hot, but action index
            # "rew": torch.from_numpy(
            #     np.array([rew * self.args.reward_scale])
            # ),  # (1,)
            "logits": logits,
            "log_prob": log_prob.view(-1),  # (1,) / scalar
            # "is_fir": torch.FloatTensor([1.0 if is_fir else 0.0]),  # (1,),
            # "done": torch.FloatTensor([1.0 if done else 0.0]),  # (1,),
            "hx": lstm_hx[0],  # (hidden,)
            "cx": lstm_hx[1],  # (hidden,)
            # "id": _id,
        }
    
    
    @staticmethod
    @rpc.functions.async_execution
    def act(learner_rref: RRef, wk_id, obs, lstm_hx):
        self = learner_rref.local_value()
        
        future = Future()
        
        def handle_result(results):
            _results = results.wait()
            self.proxy_step_data(wk_id, obs, lstm_hx, _results)
        
        future.then(handle_result) # future.wait() 결과 추가 라우팅
        future.set_result(self.infer_model.act(obs, lstm_hx))
        return future
    
    @staticmethod
    def report_data(learner_rref: RRef, wk_id, data):
        self = learner_rref.local_value()
        
        protocol, _data = decode(*data)
        if protocol is Protocol.Rollout:
            self.step_data[wk_id]["rew"] = _data["rew"]
            self.step_data[wk_id]["is_fir"] = _data["is_fir"]
            self.step_data[wk_id]["done"] = _data["done"]
            self.step_data[wk_id]["id"] = _data["id"]
            
            self.data_q.append((protocol, self.step_data[wk_id]))
    
            self.step_data[wk_id] = {} # transition 리셋
        else:
            assert protocol is Protocol.Stat
            self.data_q.append((protocol, _data))
            
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
                self.nda_obs_batch.shape[0] / (sq * sha[0])
            )  # TODO: 좋은 코드는 아닌 듯..
            idx = slice(None)  # ":"와 동일
            num_buf = bn

            return func(self, idx, num_buf)

        return _wrap

    @sample_wrapper
    def sample_batch_from_np_memory(self, idx, num_buf):
        sq = self.args.seq_len
        # bn = self.args.batch_size
        sha = self.obs_shape
        hs = self.args.hidden_size
        ac = self.args.action_space

        # (batch, seq, feat)
        nda_obs_bat = self.nda_obs_batch.reshape((num_buf, sq, *sha))[idx]
        nda_act_bat = self.nda_act_batch.reshape((num_buf, sq, 1))[idx]
        nda_rew_bat = self.nda_rew_batch.reshape((num_buf, sq, 1))[idx]
        nda_logits_bat = self.nda_logits_batch.reshape((num_buf, sq, ac))[idx]
        nda_log_prob_bat = self.nda_log_prob_batch.reshape((num_buf, sq, 1))[idx]
        nda_is_fir_bat = self.nda_is_fir_batch.reshape((num_buf, sq, 1))[idx]
        nda_hx_bat = self.nda_hx_batch.reshape((num_buf, sq, hs))[idx]
        nda_cx_bat = self.nda_cx_batch.reshape((num_buf, sq, hs))[idx]

        return (
            to_torch(nda_obs_bat),
            to_torch(nda_act_bat),
            to_torch(nda_rew_bat),
            to_torch(nda_logits_bat),
            to_torch(nda_log_prob_bat),
            to_torch(nda_is_fir_bat),
            to_torch(nda_hx_bat),
            to_torch(nda_cx_bat),
        )

    @ppo_awrapper(timer=timer)
    def learning_ppo(self): ...

    def is_bat_ready(self):
        bn = self.args.batch_size
        val = self.nda_data_num.item()
        return True if val >= bn else False

    async def put_batch_to_batch_q(self):
        while not self.stop_event.is_set():
            if self.is_bat_ready():
                batch_args = self.sample_batch_from_np_memory(
                    sampling_method="on-policy"
                )
                await self.batch_queue.put(batch_args)
                self.reset_data_num()  # 메모리 저장 인덱스 (batch_num) 초기화
                print("batch is ready !")

            await asyncio.sleep(0.001)

    async def learning_chain_ppo(self):
        self.batch_queue = asyncio.Queue(1024)
        tasks = [
            asyncio.create_task(self.learning_ppo()),
            asyncio.create_task(self.put_batch_to_batch_q()),
        ]
        await asyncio.gather(*tasks)