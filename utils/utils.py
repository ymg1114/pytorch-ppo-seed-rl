import os, sys
import cv2
import json
import psutil
import torch
import time
import platform
import pickle
import blosc2
import numpy as np
import asyncio

# import torchvision.transforms as T

from collections import deque, defaultdict
from contextlib import contextmanager
from enum import Enum, auto
from pathlib import Path

from datetime import datetime
from signal import SIGTERM  # or SIGKILL
from types import SimpleNamespace

import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote

utils = os.path.join(os.getcwd(), "utils", "parameters.json")
with open(utils) as f:
    _p = json.load(f)
    Params = SimpleNamespace(**_p)


WORKER_NAME = "Worker{}"
LEARNER_NAME = "Learner"


def call_method(method, rref, *args, **kwargs):
    r"""
    a helper function to call a method on the given RRef
    """

    return method(rref.local_value(), *args, **kwargs)


def remote_method(method, rref, *args, **kwargs):
    r"""
    a helper function to run method on the owner of rref and fetch back the
    result using RPC
    """
    
    args = [method, rref] + list(args)
    print(args)
    print(kwargs)
    return rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)


def remote_method_async(method, rref, *args, **kwargs):
    """
    A helper function to asynchronously run a method on the owner of rref and
    fetch back the result using RPC.
    """
    
    # Prepare the arguments for the remote call
    args = [method, rref] + list(args)
    # Asynchronously call the `call_method` function on the remote node
    return rpc.rpc_async(rref.owner(), call_method, args=args, kwargs=kwargs)


async def to_asyncio_future(torch_future):
    """Convert a PyTorch Future to an asyncio Future.
    
    Example Usages)
    
    async def main():
        # Assuming rpc is properly initialized and there is a remote node available
        rref = rpc.remote("worker1", some_remote_class)
        
        # Asynchronously call a method and await its result
        torch_future = remote_method_async(some_method, rref, arg1, arg2)
        result = await to_asyncio_future(torch_future)
        
        print(f"Result from remote method: {result}")
        
    # Initialize RPC, run the main function, and then shut down RPC
    if __name__ == "__main__":
        rpc.init_rpc("worker0", rank=0, world_size=2)
        asyncio.run(main())
        rpc.shutdown()
    """
    
    loop = asyncio.get_event_loop()
    asyncio_future = loop.create_future()

    def copy_result(future):
        if future.exception():
            loop.call_soon_threadsafe(asyncio_future.set_exception, future.exception())
        else:
            loop.call_soon_threadsafe(asyncio_future.set_result, future.wait())

    torch_future.then(copy_result)
    return asyncio_future


class SingletonMetaCls(type):
    __instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls.__instances:
            cls.__instances[cls] = super().__call__(*args, **kwargs)
        return cls.__instances[cls]


# TODO: 이런 하드코딩 스타일은 바람직하지 않음. 더 좋은 코드 구조로 개선 필요.
DataFrameKeyword = [
    "obs_batch",
    "act_batch",
    "rew_batch",
    "logits_batch",
    "log_prob_batch",
    "is_fir_batch",
    "hx_batch",
    "cx_batch",
    "batch_num",
]


dt_string = datetime.now().strftime(f"[%d][%m][%Y]-%H_%M")
result_dir = os.path.join("results", str(dt_string))
model_dir = os.path.join(result_dir, "models")


ErrorComment = "Should be PPO"


# Centralized One Master Node IP/PORT
MASTER_ADDR = "localhost"
MASTER_PORT = Params.master_port


flatten = lambda obj: obj.numpy().reshape(-1).astype(np.float32)


to_torch = lambda nparray: torch.from_numpy(nparray).type(torch.float32)


to_cpu_tensor = lambda *data: tuple(d.cpu() for d in data)


def extract_file_num(filename):
    parts = filename.stem.split("_")
    try:
        return int(parts[-1])
    except ValueError:
        return -1


def make_gpu_batch(*args, device):
    to_gpu = lambda tensor: tensor.to(device)
    return tuple(map(to_gpu, args))


def get_current_process_id():
    return os.getpid()


def get_process_id(name):
    for proc in psutil.process_iter(["pid", "name"]):
        if proc.info["name"] == name:
            return proc.info["pid"]


def kill_process(pid):
    os.kill(pid, SIGTERM)


class KillSubProcesses:
    def __init__(self, processes):
        self.target_processes = processes
        self.os_system = platform.system()

    def __call__(self):
        assert hasattr(self, "target_processes")
        for p in self.target_processes:
            if self.os_system == "Windows":
                print("This is a Windows operating system.")
                p.terminate()  # Windows에서는 "관리자 권한" 이 필요.

            elif self.os_system == "Linux":
                print("This is a Linux operating system.")
                os.kill(p.pid, SIGTERM)


def mul(shape_dim):
    _val = 1
    for e in shape_dim:
        _val *= e
    return _val


def counted(f):
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return f(*args, **kwargs)

    wrapper.calls = 0
    return wrapper


class ExecutionTimer:
    def __init__(self, threshold=100, num_transition=None):
        self.num_transition = num_transition
        self.timer_dict = defaultdict(lambda: deque(maxlen=threshold))
        self.throughput_dict = defaultdict(lambda: deque(maxlen=threshold))

    @contextmanager
    def timer(self, code_block_name: str, check_throughput=False):
        start_time = time.time()
        yield  # 사용자가 지정 코드 블록이 실행되는 부분
        end_time = time.time()

        elapsed_time = end_time - start_time

        self.timer_dict[code_block_name].append(elapsed_time)  # sec
        if self.num_transition is not None and isinstance(
            self.num_transition, (int, float, np.number)
        ):
            if check_throughput is True:
                self.throughput_dict[code_block_name].append(
                    self.num_transition / (elapsed_time + 1e-6)
                )  # transition/sec
        # avg_time = sum(self.exec_times) / len(self.exec_times)


def SaveErrorLog(error: str, log_dir: str):
    current_time = time.strftime("[%Y_%m_%d][%H_%M_%S]", time.localtime(time.time()))
    # log_dst = os.path.join(log_dir, f"error_log_{current_time}.txt")
    dir = Path(log_dir)
    error_log = dir / f"error_log_{current_time}.txt"
    error_log.write_text(f"{error}\n")
    return


class Protocol(Enum):
    Rollout = auto()
    Stat = auto()


def KillProcesses(pid):
    parent = psutil.Process(pid)
    for child in parent.children(
        recursive=True
    ):  # or parent.children() for recursive=False
        child.kill()
    parent.kill()


def encode(protocol, data):
    return pickle.dumps(protocol), blosc2.compress(pickle.dumps(data), clevel=1)


def decode(protocol, data):
    return pickle.loads(protocol), pickle.loads(blosc2.decompress(data))


if __name__ == "__main__":
    KillProcesses()
