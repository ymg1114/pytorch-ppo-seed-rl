import asyncio
import copy
import numpy as np
import threading as th

from collections import deque, defaultdict
from typing import List
from functools import partial

import torch
import torch.jit as jit
from torch.futures import Future
from torch.optim import Adam, RMSprop

from networks.models import MlpLSTM

from utils.utils import (
    set_model_weight,
    Protocol,
    make_gpu_batch,
    ExecutionTimer,
    Params,
    to_torch,
    counted,
)
from env.proxy import EnvSpace

from agents.learner_storage import LearnerStorage

from buffers import grpc_service_pb2 as Pb2
from buffers import grpc_service_pb2_grpc as Pb2gRPC

from .storage_module.batch_memory import NdaMemInterFace
from . import (
    ppo_awrapper,
)

from utils.utils import serialize, deserialize

timer = ExecutionTimer(
    num_transition=Params.seq_len * Params.batch_size * 1
)  # Learner에서 데이터 처리량 (학습)


class LearnerRPC(Pb2gRPC.RunnerServiceServicer, NdaMemInterFace):
    def __init__(
        self,
        args,
        nda_ref,
        obs_shape,
    ):
        NdaMemInterFace.__init__(self, nda_ref=nda_ref)
        self.get_nda_memory_interface()

        self.args = args
        self.device = self.args.device
        self.obs_shape = obs_shape

        self.stop_event = th.Event()  # condition
        self.step_data = defaultdict(dict)
        self.data_q = deque(maxlen=1024)  # rollout queue

        self.stat_log_idx = 0
        self.stat_log_cycle = 50
        self.stat_q = deque(maxlen=self.stat_log_cycle)  # stat queue
        self.lock = th.Lock()  # Lock 객체 생성

        self.model = jit.script(MlpLSTM(args=Params, env_space=EnvSpace)).to(
            self.device
        )
        # self.model = MlpLSTM(args=Params, env_space=EnvSpace).to(self.device)
        learner_model_state_dict = set_model_weight(self.args)
        if learner_model_state_dict is not None:
            self.model.load_state_dict(learner_model_state_dict)

        self.infer_model = copy.deepcopy(self.model)
        self.infer_model.eval()

        self.to_gpu = partial(make_gpu_batch, device=self.device)
        self.optimizer = RMSprop(self.model.parameters(), lr=self.args.lr, eps=1e-5)

        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=args.result_dir)

    def run(self):
        self.child_thread = []

        storage = LearnerStorage(
            self.args,
            self.nda_ref,
            self.stop_event,
            self.obs_shape,
        )
        t1 = th.Thread(
            target=storage.entry_chain,
            args=(
                storage,
                self.data_q,
            ),
            daemon=True,
        )
        t2 = th.Thread(target=self.entry_chain, daemon=True)
        self.child_thread.extend([t1, t2])

        for ct in self.child_thread:
            ct.start()

    def join(self):
        for ct in self.child_thread:
            ct.join()

    def entry_chain(self):
        asyncio.run(self.learning_chain_ppo())

    def Act(self, request, context):
        # 각 워커 클라이언트로부터 받은 데이터를 역직렬화
        obs = deserialize(request.obs)
        lstm_hx = deserialize(request.lstm_hx)
        id_ = request.id

        # 러너 서버 측에서 행동 출력 (GPU 인퍼런스)
        act, logits, log_prob, lstm_hx = self.infer_model.act(
            obs.to(self.device),
            (lstm_hx[0].to(self.device), lstm_hx[1].to(self.device)),
        )

        with self.lock:  # Critical Section 보호
            self.step_data[id_] = {
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

        # 응답 메시지 생성
        act_response = Pb2.ActResponse(
            act=serialize(act),
            logits=serialize(logits),
            log_prob=serialize(log_prob),
            lstm_hx=serialize(lstm_hx),
        )
        return act_response

    def Report(self, request, context):
        # ReportRequest의 oneof 필드를 처리
        if request.HasField("data_request"):
            data_request = request.data_request
            rew = data_request.rew
            is_fir = data_request.is_fir
            done = data_request.done
            id_ = data_request.id

            with self.lock:  # Critical Section 보호
                self.step_data[id_]["rew"] = torch.from_numpy(np.array([rew]))
                self.step_data[id_]["is_fir"] = torch.tensor([1.0 if is_fir else 0.0])
                self.step_data[id_]["done"] = torch.tensor([1.0 if done else 0.0])
                self.step_data[id_]["id"] = id_

                self.data_q.append((Protocol.Rollout, self.step_data[id_]))
                self.step_data[id_] = {}  # transition 리셋

        elif request.HasField("stat_request"):
            stat_request = request.stat_request
            epi_rew = stat_request.epi_rew
            id_ = stat_request.id

            with self.lock:  # Critical Section 보호
                self.stat_q.append(epi_rew)

                if (
                    len(self.stat_q) > 0
                    and self.stat_log_idx % self.stat_log_cycle == 0
                ):
                    self.log_stat(
                        {
                            "log_len": len(self.stat_q),
                            "mean_stat": np.mean([self.stat_q]),
                        }
                    )
                    self.stat_log_idx = 0
                self.stat_log_idx += 1
        else:
            assert False, f"Wrong gRPC request from Client -> Server: {request}"
        return Pb2.Empty()

    @counted
    def log_stat(self, data):
        _len = data["log_len"]
        _epi_rew = data["mean_stat"]

        tag = f"worker/{_len}-game-mean-stat-of-epi-rew"
        x = self.log_stat.calls * _len  # global game counts
        y = _epi_rew

        print(f"tag: {tag}, y: {y}, x: {x}")

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

    def log_loss_tensorboard(self, timer: ExecutionTimer, loss, detached_losses):
        self.writer.add_scalar("total-loss", float(loss.item()), self.idx)

        if "value-loss" in detached_losses:
            self.writer.add_scalar(
                "original-value-loss", detached_losses["value-loss"], self.idx
            )

        if "policy-loss" in detached_losses:
            self.writer.add_scalar(
                "original-policy-loss", detached_losses["policy-loss"], self.idx
            )

        if "policy-entropy" in detached_losses:
            self.writer.add_scalar(
                "original-policy-entropy", detached_losses["policy-entropy"], self.idx
            )

        if "ratio" in detached_losses:
            self.writer.add_scalar(
                "min-ratio", detached_losses["ratio"].min(), self.idx
            )
            self.writer.add_scalar(
                "max-ratio", detached_losses["ratio"].max(), self.idx
            )
            self.writer.add_scalar(
                "avg-ratio", detached_losses["ratio"].mean(), self.idx
            )

        if timer is not None and isinstance(timer, ExecutionTimer):
            for k, v in timer.timer_dict.items():
                self.writer.add_scalar(
                    f"{k}-elapsed-mean-sec", sum(v) / (len(v) + 1e-6), self.idx
                )
            for k, v in timer.throughput_dict.items():
                self.writer.add_scalar(
                    f"{k}-transition-per-secs", sum(v) / (len(v) + 1e-6), self.idx
                )

        if len(self.stat_q) >= self.stat_log_cycle:
            self.writer.add_scalar(
                f"worker/{len(self.stat_q)}-game-mean-stat-of-epi-rew",
                np.mean([self.stat_q]),
                self.idx,
            )
