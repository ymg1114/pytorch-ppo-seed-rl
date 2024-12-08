import os, sys
import gc
import gym
import asyncio
import traceback

# import grpc
from grpc.aio import server as aio_server # 비동기 서버 생성
import torch
import multiprocessing as mp

from concurrent import futures
from types import SimpleNamespace as SN
from pathlib import Path

from agents.storage_module.batch_memory import setup_nda_memory
from agents.worker import WorkerRPC
from agents.learner import (
    LearnerRPC,
)

from buffers import grpc_service_pb2 as Pb2
from buffers import grpc_service_pb2_grpc as Pb2gRPC

from utils.utils import (
    KillProcesses,
    Params,
    result_dir,
    model_dir,
    # LEARNER_NAME,
    # WORKER_NAME,
    # SERVER_IP,
    SERVER_PORT,
)


fn_dict = {}


def register(fn):
    fn_dict[fn.__name__] = fn
    return fn


class Runner:
    def __init__(self):
        mp.set_start_method("spawn")

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

    @staticmethod
    async def stop_server(server):
        """gRPC 서버 종료 메서드"""
        
        await server.shutdown()
        await server.wait_for_termination()

    @staticmethod
    async def start_learner_proxy(args, obs_shape):
        """학습 프로세스와 서버 실행"""
        
        stop_event = asyncio.Event()

        # 학습을 위한 공유 메모리 확보
        nda_ref = setup_nda_memory(
            args, obs_shape, batch_size=args.batch_size
        )

        learner = LearnerRPC(
            args,
            nda_ref,
            stop_event,
            obs_shape,
        )

        # gRPC 서버 초기화
        aserver = aio_server()
        Pb2gRPC.add_RunnerServiceServicer_to_server(learner, aserver)
        aserver.add_insecure_port(f"[::]:{SERVER_PORT}")

        try:
            # 비동기 작업 실행
            await asyncio.gather(
                aserver.start(),
                learner.run_async(),
                aserver.wait_for_termination(),
            )
        except asyncio.CancelledError:
            print("Server tasks cancelled.")
        except Exception as e:
            print(f"Error in learner proxy: {e}")
            traceback.print_exc()
        finally:
            await Runner.stop_server(aserver)  # 명시적으로 서버 종료

    @staticmethod
    def start_learner(args, obs_shape):
        """동기식 인터페이스에서 비동기 서버 실행"""
        
        asyncio.run(Runner.start_learner_proxy(args, obs_shape))

    @register
    def run_server(self):
        """서버 실행 진입점"""
        
        try:
            Runner.start_learner(self.args, self.obs_shape)
        except Exception as e:
            print(f"Error in server: {e}")
            traceback.print_exc(limit=128)
        finally:
            gc.collect()  # Force garbage collection

    @staticmethod
    def start_worker(args, worker_id):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # 워커 객체를 프로세스 내부에서 생성
            worker = WorkerRPC(args, f"Worker-{worker_id}")
            loop.run_until_complete(worker.gene_rollout())
        except Exception as e:
            print(f"Error in Worker-{worker_id}: {e}")
            traceback.print_exc()
        finally:
            loop.close()

    @register
    def run_client(self):
        child_process = {}

        try:
            for rank in range(self.args.num_worker):
                p = mp.Process(target=Runner.start_worker, args=(self.args, rank))
                child_process[rank] = p

            for r, p in child_process.items():
                p.start()

            for r, p in child_process.items():
                p.join()

        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc(limit=128)

            for r, p in child_process.items():
                if p.is_alive():
                    p.terminate()
                    p.join()
        finally:
            KillProcesses(os.getpid())
            gc.collect()  # Force garbage collection

    def run(self):
        assert len(sys.argv) == 2
        func_name = sys.argv[1]

        if func_name in fn_dict:
            fn_dict[func_name](self)
        else:
            assert False, f"Wrong func_name: {func_name}"


if __name__ == "__main__":
    rn = Runner()
    rn.run()
