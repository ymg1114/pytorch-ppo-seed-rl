import asyncio
import copy
import time
import grpc
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
        stop_event,
        obs_shape,
    ):
        NdaMemInterFace.__init__(self, nda_ref=nda_ref)
        self.get_nda_memory_interface()

        self.args = args
        self.num_worker = args.num_worker # 개별 클라이언트 프로세스 개수
        self.device = self.args.device
        self.obs_shape = obs_shape

        self.stop_event = stop_event  # condition
        self.step_data = defaultdict(dict)

        self.stat_log_idx = 0
        self.stat_log_cycle = 50
        
        self.batch_event = asyncio.Event()
        self.lock = asyncio.Lock()  # Lock 객체 생성

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

        # processing: 요청이 처리 중인 상태인지 확인
        # request: 클라이언트 요청 데이터 저장
        # response: 클라이언트 응답 데이터 저장
        self.client_states = defaultdict(lambda: {"request": None, "response": None, "processing": False})

        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=args.result_dir)

    # TODO: 클라이언트 사이드 보다, 러너 (서버) 사이드의 프로그램을 먼저 실행시키고, 웜업 시간을 잠시 가져야 함
    async def run_async(self):
        self.batch_queue = asyncio.Queue(1024)
        self.data_q = asyncio.Queue(maxsize=1024)  # rollout queue
        self.stat_q = asyncio.Queue(maxsize=self.stat_log_cycle)  # stat queue

        storage = LearnerStorage(
            self.args,
            self.nda_ref,
            self.stop_event,
            self.obs_shape,
        )

        tasks = [
            asyncio.create_task(storage.memory_chain(self.data_q)),
            asyncio.create_task(self.learning_ppo()),
            asyncio.create_task(self.put_batch_to_batch_q()),
            asyncio.create_task(self.batch_processor()),
        ]
        await asyncio.gather(*tasks)
        
    async def Act(self, request, context):
        client_id = request.id

        # 요청 동기화: 중복 요청 방지
        async with self.lock:
            c_status = self.client_states[client_id]
            assert "request" in c_status
            assert "response" in c_status
            assert "processing" in c_status

            if c_status["processing"]:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(f"Client {client_id} already has a pending request.")
                raise RuntimeError(f"Client {client_id} has a pending request.")
                # return Pb2.ActResponse()  # 빈 응답

            # 요청 저장 및 상태 갱신
            c_status["request"] = request
            c_status["processing"] = True  # 처리 상태 진입

            # 배치 처리 준비 상태 확인
            if sum(status["processing"] for status in self.client_states.values()) >= self.num_worker:
                self.batch_event.set()

        # 응답 대기
        while not c_status["response"]:
            await asyncio.sleep(1e-5)

        # 응답 반환
        async with self.lock:
            res = c_status["response"]
            c_status["response"] = None  # 응답 초기화
            c_status["processing"] = False  # 처리 상태 탈출

        return res

    async def batch_processor(self):
        while True:
            # 배치 크기만큼 요청 대기
            await self.batch_event.wait()

            # 배치 처리
            async with self.lock:  # Critical Section 보호
                # 요청 데이터 수집 및 변환
                batch, client_ids = [], []
                for client_id, status in self.client_states.items():
                    request = status["request"]
                    obs = deserialize(request.obs)
                    lstm_hx = deserialize(request.lstm_hx)
                    batch.append((obs, lstm_hx))
                    client_ids.append(client_id)

                # 배치 데이터 준비
                obs_batch = torch.stack([obs.to(self.device) for obs, _ in batch])
                lstm_hx_batch = (
                    torch.stack([hx[0].to(self.device) for _, hx in batch]),
                    torch.stack([hx[1].to(self.device) for _, hx in batch]),
                )

                # 배치 inference 수행 (GPU 인퍼런스)
                acts, logits, log_probs, lstm_hxs = self.infer_model.act(obs_batch, lstm_hx_batch)

                # 배치 응답 메시지 생성 및 저장
                for idx, (act, logit, log_prob, lstm_hx) in enumerate(
                    zip(acts, logits, log_probs, zip(*lstm_hxs))
                ):
                    client_id = client_ids[idx]
                    obs, _ = batch[idx]

                    # 클라이언트 상태 데이터 저장
                    self.step_data[client_id] = {
                        "obs": obs, # (c, h, w) or (D,)
                        "act": act.view(-1), # (1,)
                        "logits": logit,
                        "log_prob": log_prob.view(-1), # (1,)
                        "hx": lstm_hx[0], # (hidden,)
                        "cx": lstm_hx[1], # (hidden,) 
                    }

                    # 클라이언트 응답 준비
                    self.client_states[client_id]["response"] = Pb2.ActResponse(
                        act=serialize(act),
                        logits=serialize(logit),
                        log_prob=serialize(log_prob),
                        lstm_hx=serialize(lstm_hx),
                    )

                    self.client_states[client_id]["request"] = None  # 요청 초기화

                # 배치 이벤트 초기화
                self.batch_event.clear()

            await asyncio.sleep(1e-5)

    async def Report(self, request, context):
        # ReportRequest의 oneof 필드를 처리
        async with self.lock:  # Critical Section 보호
            if request.HasField("data_request"):
                data_request = request.data_request
                rew = data_request.rew
                is_fir = data_request.is_fir
                done = data_request.done
                client_id = data_request.id

                self.step_data[client_id]["rew"] = torch.from_numpy(np.array([rew]))
                self.step_data[client_id]["is_fir"] = torch.tensor([1.0 if is_fir else 0.0])
                self.step_data[client_id]["done"] = torch.tensor([1.0 if done else 0.0])
                self.step_data[client_id]["id"] = client_id
                
                await self.data_q.put((Protocol.Rollout, self.step_data[client_id]))
                self.step_data[client_id] = {}  # transition 리셋

            elif request.HasField("stat_request"):
                stat_request = request.stat_request
                epi_rew = stat_request.epi_rew
                client_id = stat_request.id

                await self.process_statistics(epi_rew)

                # 해당 경기 리스트 플레이스홀더 제거
                self.step_data.pop(client_id)
                self.client_states.pop(client_id)
            else:
                assert False, f"Wrong gRPC request from Client -> Server: {request}"
        return Pb2.Empty()

    async def process_statistics(self, epi_rew):
        # 통계 데이터를 큐에 추가
        await self.stat_q.put(epi_rew)

        if (
            self.stat_q.qsize() > 0  # 큐에 데이터가 있는지 확인
            and self.stat_log_idx % self.stat_log_cycle == 0
        ):
            # 큐에서 모든 데이터를 꺼내 평균 계산
            stats = []
            while not self.stat_q.empty():
                stats.append(await self.stat_q.get())

            # 로그 작성
            self.log_stat(
                {
                    "log_len": len(stats),
                    "mean_stat": np.mean(stats),
                }
            )
            self.stat_log_idx = 0
        self.stat_log_idx += 1

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

    async def log_loss_tensorboard(self, timer: ExecutionTimer, loss, detached_losses):
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

        if self.stat_q.qsize() >= self.stat_log_cycle:
            stats = []
            # 큐에서 데이터를 꺼내 리스트로 수집
            while not self.stat_q.empty():
                stats.append(await self.stat_q.get())

            # 평균값 계산 후 TensorBoard에 기록
            self.writer.add_scalar(
                f"worker/{len(stats)}-game-mean-stat-of-epi-rew",
                np.mean(stats),
                self.idx,
            )