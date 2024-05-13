import os
import gym
import traceback

import torch
import torch.multiprocessing as mp
# import multiprocessing as mp
import torch.distributed as dist
import torch.distributed.rpc as rpc

from types import SimpleNamespace as SN
from pathlib import Path

from agents.storage_module.shared_batch import (
    reset_shared_on_policy_memory,
)
from agents.learner import (
    LearnerRPC,
)
from networks.models import (
    MlpLSTMBase,
)
from utils.utils import (
    KillProcesses,
    Params,
    result_dir,
    model_dir,
    extract_file_num,
    ErrorComment,
    LEARNER_NAME,
    WORKER_NAME, 
    MASTER_ADDR, 
    MASTER_PORT,
)


class Runner:
    def __init__(self):
        mp.set_start_method('spawn')
        
        self.args = Params
        self.args.device = torch.device(
            f"cuda:{Params.gpu_idx}" if torch.cuda.is_available() else "cpu"
        )

        # 미리 정해진 경로가 있다면 그것을 사용
        self.args.result_dir = self.args.result_dir or result_dir
        self.args.model_dir = self.args.model_dir or model_dir
        print(f"device: {self.args.device}")

        _model_dir = Path(self.args.model_dir)
        # 경로가 디렉토리인지 확인
        if not _model_dir.is_dir():
            # 디렉토리가 아니라면 디렉토리 생성
            _model_dir.mkdir(parents=True, exist_ok=True)

        # only to get observation, action space
        env = gym.make(self.args.env)
        # env.seed(0)
        self.n_outputs = (
            env.action_space.n
            if hasattr(env.action_space, "n")
            else env.action_space.shape[0]
        )
        self.args.action_space = self.n_outputs
        print("Action Space: ", self.n_outputs)
        print("Observation Space: ", env.observation_space.shape)

        # 이산 행동 분포 환경: openai-gym의 "CartPole-v1"
        assert len(env.observation_space.shape) <= 1

        self.stop_event = mp.Event()

        module_switcher = {  # (learner_cls, model_cls)
            "PPO": SN(learner_cls=LearnerRPC, model_cls=MlpLSTMBase),
        }
        module_name_space = module_switcher.get(
            self.args.algo, lambda: AssertionError(ErrorComment)
        )

        self.LearnerCls = module_name_space.learner_cls
        self.ModelCls = module_name_space.model_cls

        self.obs_shape = [env.observation_space.shape[0]]
        env.close()

        # num of Nodes
        self.world_size = self.args.num_worker + 1 # workers, learner
        
        self.Model = self.ModelCls(
            *self.obs_shape, self.n_outputs, self.args.seq_len, self.args.hidden_size
        )
        
        learner_model_state_dict = self.set_model_weight(self.args.model_dir)
        if learner_model_state_dict is not None:
            self.Model.load_state_dict(learner_model_state_dict)
            
        self.Model.to(torch.device("cpu"))  # cpu 모델

    def set_model_weight(self, model_dir):
        model_files = list(Path(model_dir).glob(f"{self.args.algo}_*.pt"))

        prev_model_weight = None
        if len(model_files) > 0:
            sorted_files = sorted(model_files, key=extract_file_num)
            if sorted_files:
                prev_model_weight = torch.load(
                    sorted_files[-1],
                    map_location=torch.device("cpu"),  # 가장 최신 학습 모델 로드
                )

        learner_model_state_dict = self.Model.cpu().state_dict()
        if prev_model_weight is not None:
            learner_model_state_dict = {
                k: v.cpu() for k, v in prev_model_weight.state_dict().items()
            }

        return learner_model_state_dict  # cpu 텐서

    def _start(self, rank):
        os.environ['MASTER_ADDR'] = MASTER_ADDR
        os.environ['MASTER_PORT'] = str(MASTER_PORT)
        
        # rank0 is the learner
        if rank == 0:
            learner_name = LEARNER_NAME
            options = rpc.TensorPipeRpcBackendOptions()
            
            device_map = {torch.device("cpu"): torch.device(self.args.device)} # {src: dst} / {local: remote}
            options.set_device_map(learner_name, device_map)
            
            rpc.init_rpc(learner_name, rank=rank, world_size=self.world_size, rpc_backend_options=options)

            # 학습을 위한 공유메모리 확보
            shm_ref_switcher = {
                "PPO": reset_shared_on_policy_memory,
            }
            shm_ref_factory = shm_ref_switcher.get(
                self.args.algo, lambda: AssertionError(ErrorComment)
            )
            shm_ref = shm_ref_factory(self.args, self.obs_shape)
            
            learner = self.LearnerCls(self.args, self.Model, shm_ref, self.stop_event, self.obs_shape)
            learner.run()
            
        # workers passively waiting for instructions from learner
        else:
            remote_worker_name = WORKER_NAME.format(rank)
            options = rpc.TensorPipeRpcBackendOptions()
            
            device_map = {torch.device(self.args.device): torch.device("cpu")} # {src: dst} / {local: remote}
            options.set_device_map(remote_worker_name, device_map)
            
            rpc.init_rpc(remote_worker_name, rank=rank, world_size=self.world_size, rpc_backend_options=options)

        rpc.shutdown()


    def start(self):
        child_process = {}

        try:            
            for rank in range(self.world_size):
                p = mp.Process(target=self._start, args=(rank,))
                child_process[rank] = p

            for r, p in child_process.items():
                p.start()

            for r, p in child_process.items():
                p.join()

        except Exception as e:
            # 자식 프로세스 종료 신호 보냄
            self.stop_event.set()

            print(f"error: {e}")
            traceback.print_exc(limit=128)

            for r, p in child_process.items():
                if p.is_alive():
                    p.terminate()
                    p.join()
        finally:
            KillProcesses(os.getpid())


if __name__ == "__main__":
    rn = Runner()
    rn.start()
