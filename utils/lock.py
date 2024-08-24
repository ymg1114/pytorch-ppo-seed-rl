from contextlib import contextmanager

import queue
import torch.multiprocessing as mp

# import multiprocessing as mp


class Lock:
    def __init__(self):
        self._lock = mp.Lock()

    # 여러 프로세스가 공유 자원에 동시에 접근하는 것을 방지하여 데이터 무결성을 보장
    @contextmanager
    def lock(self):
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()
        return

    def get(self, q: mp.Queue):
        with self.lock():
            data = q.get()
        return data

    def put(self, q: mp.Queue, data):
        with self.lock():
            q.put(data)
        return
