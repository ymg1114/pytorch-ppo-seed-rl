import os, sys
import gc
import gym
import traceback

import torch
import torch.multiprocessing as mp
# import multiprocessing as mp
import torch.distributed as dist
import torch.distributed.rpc as rpc

from types import SimpleNamespace as SN
from pathlib import Path

from agents.storage_module.batch_memory import setup_nda_memory
from agents.learner import (
    LearnerRPC,
)

from utils.utils import (
    KillProcesses,
    Params,
    result_dir,
    model_dir,
    LEARNER_NAME,
    WORKER_NAME, 
    MASTER_ADDR, 
    MASTER_PORT,
)


fn_dict = {}


def register(fn):
    fn_dict[fn.__name__] = fn
    return fn


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
        
        self.obs_shape = [env.observation_space.shape[0]]
        env.close()

        # num of Nodes
        self.world_size = self.args.num_worker + 1 # workers, learner

    def _start(self, rank):
        os.environ['MASTER_ADDR'] = MASTER_ADDR
        os.environ['MASTER_PORT'] = MASTER_PORT
        
        # rank0 is the learner
        if rank == 0:
            rpc.init_rpc(LEARNER_NAME, rank=rank, world_size=self.world_size)
            
            # 학습을 위한 메모리 확보
            nda_ref = setup_nda_memory(self.args, self.obs_shape, batch_size=self.args.batch_size)

            learner = LearnerRPC(self.args, nda_ref, self.obs_shape)
            learner.run()
            
        # workers passively waiting for instructions from learner
        else:
            remote_worker_name = WORKER_NAME.format(rank)
            options = rpc.TensorPipeRpcBackendOptions(
                num_worker_threads=4,
            )
            device_map = {torch.device("cpu"): torch.device(self.args.device)} # {src-node: callee-node}
            options.set_device_map(LEARNER_NAME, device_map)
            
            rpc.init_rpc(remote_worker_name, rank=rank, world_size=self.world_size, rpc_backend_options=options)

        # block until all rpcs finish
        rpc.shutdown()

    @register
    def run_master(self):
        try:
            self._start(rank=0) # rank of learner node is "0"
            
        except Exception as e:
            print(f"error: {e}")
            traceback.print_exc(limit=128)

        finally:
            rpc.shutdown() # Clean up RPC resources
            gc.collect() # Force garbage collection
            
    @register
    def run_slave(self):
        child_process = {}

        try:
            # rank of worker node is "1~"
            for rank in range(1, self.world_size):
                p = mp.Process(target=self._start, args=(rank,))
                child_process[rank] = p

            for r, p in child_process.items():
                p.start()

            for r, p in child_process.items():
                p.join()

        except Exception as e:
            print(f"error: {e}")
            traceback.print_exc(limit=128)

            for r, p in child_process.items():
                if p.is_alive():
                    p.terminate()
                    p.join()
        finally:
            KillProcesses(os.getpid())
            rpc.shutdown() # Clean up RPC resources
            gc.collect() # Force garbage collection
            
    def start(self):
        assert len(sys.argv) == 2
        func_name = sys.argv[1]
        
        if func_name in fn_dict:
            fn_dict[func_name](self)
        else:
            assert False, f"Wrong func_name: {func_name}"
        

if __name__ == "__main__":
    rn = Runner()
    rn.start()
