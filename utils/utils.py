import os, sys
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


utils = os.path.join(os.getcwd(), "utils", "parameters.json")
with open(utils) as f:
    _p = json.load(f)
    Params = SimpleNamespace(**_p)


WORKER_NAME = "Worker{}"
LEARNER_NAME = "Learner"


def set_model_weight(args):
    model_files = list(Path(args.model_dir).glob(f"{args.algo}_*.pt"))

    prev_model_weight = None
    if len(model_files) > 0:
        sorted_files = sorted(model_files, key=extract_file_num)
        if sorted_files:
            prev_model_weight = torch.load(
                sorted_files[-1],
                map_location=torch.device("cpu"),  # 가장 최신 학습 모델 로드
            )

    if prev_model_weight is not None:
        return {
            k: v.cpu() for k, v in prev_model_weight.state_dict().items()
        }  # cpu 텐서


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


# Centralized One Learner Server IP/PORT
SERVER_IP = "localhost"
SERVER_PORT = str(Params.server_port)


flatten = lambda obj: obj.numpy().reshape(-1).astype(np.float32)


to_torch = lambda nparray: torch.from_numpy(nparray).type(torch.float32)


to_cpu_tensor = lambda *data: tuple(d.cpu() for d in data)


# 직렬화 함수
def serialize(data):
    return pickle.dumps(data)


# 역직렬화 함수
def deserialize(serialized_data):
    return pickle.loads(serialized_data)


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


if __name__ == "__main__":
    KillProcesses()
