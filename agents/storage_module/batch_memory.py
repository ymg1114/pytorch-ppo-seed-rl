import numpy as np

from utils.utils import mul, DataFrameKeyword


def ndarray_memory(nda_ref: dict, np_array: np.ndarray, name: str):
    assert name in DataFrameKeyword

    nda_ref.update({name: np_array.astype(np.float32)})


def setup_nda_memory(args, obs_shape, batch_size):
    nda_ref = {}

    obs_batch = np.zeros(
        args.seq_len * batch_size * mul(obs_shape),
        dtype=np.float32,
    )  # observation-space
    ndarray_memory(nda_ref, obs_batch, "obs_batch")

    act_batch = np.zeros(
        args.seq_len * batch_size * 1, dtype=np.float32
    )  # not one-hot, but action index (scalar)
    ndarray_memory(nda_ref, act_batch, "act_batch")

    rew_batch = np.zeros(args.seq_len * batch_size * 1, dtype=np.float32)  # scalar
    ndarray_memory(nda_ref, rew_batch, "rew_batch")

    logits_batch = np.zeros(
        args.seq_len * batch_size * args.action_space,
        dtype=np.float32,
    )  # action-space (logits)
    ndarray_memory(nda_ref, logits_batch, "logits_batch")

    log_prob_batch = np.zeros(args.seq_len * batch_size * 1, dtype=np.float32)  # scalar
    ndarray_memory(nda_ref, log_prob_batch, "log_prob_batch")

    is_fir_batch = np.zeros(args.seq_len * batch_size * 1, dtype=np.float32)  # scalar
    ndarray_memory(nda_ref, is_fir_batch, "is_fir_batch")

    hx_batch = np.zeros(
        args.seq_len * batch_size * args.hidden_size,
        dtype=np.float32,
    )  # hidden-states
    ndarray_memory(nda_ref, hx_batch, "hx_batch")

    cx_batch = np.zeros(
        args.seq_len * batch_size * args.hidden_size,
        dtype=np.float32,
    )  # cell-states
    ndarray_memory(nda_ref, cx_batch, "cx_batch")

    # 메모리 저장 인덱스
    nda_data_num = np.array([0], dtype=np.float32)  # 초기화
    nda_ref.update({"batch_index": nda_data_num})
    return nda_ref


class NdaMemInterFace:
    def __init__(self, nda_ref):
        self.nda_ref = nda_ref

    def get_nda_memory(self, name: str):
        assert hasattr(self, "nda_ref") and name in self.nda_ref
        return self.nda_ref.get(name)

    def get_nda_memory_interface(self):
        assert hasattr(self, "nda_ref")

        self.nda_obs_batch = self.get_nda_memory("obs_batch")
        self.nda_act_batch = self.get_nda_memory("act_batch")
        self.nda_rew_batch = self.get_nda_memory("rew_batch")
        self.nda_logits_batch = self.get_nda_memory("logits_batch")
        self.nda_log_prob_batch = self.get_nda_memory("log_prob_batch")
        self.nda_is_fir_batch = self.get_nda_memory("is_fir_batch")
        self.nda_hx_batch = self.get_nda_memory("hx_batch")
        self.nda_cx_batch = self.get_nda_memory("cx_batch")

        self.nda_data_num = self.nda_ref.get("batch_index")

        self.reset_data_num()  # 메모리 저장 인덱스 (batch_num) 초기화

    def reset_data_num(self):
        self.nda_data_num[:] = 0
