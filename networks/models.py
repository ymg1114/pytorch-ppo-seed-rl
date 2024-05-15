import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit

from torch import Tensor
from torch.distributions import Categorical

from types import SimpleNamespace as SN
from typing import Tuple


class MlpLSTM(nn.Module):
    def __init__(self, args, env_space):
        super().__init__()
        self.args = args
        self.env_space = env_space
        
        self.input_size = env_space["obs"].shape # (4,)
        self.n_outputs = env_space["act"].n # 2
        self.sequence_length = env_space["info"]["seq_len"].n # 5
        self.hidden_size = env_space["info"]["hidden_size"].n # 64
        self.batch_size = env_space["info"]["batch_size"].n # 128
        
        self.body = nn.Sequential(
            nn.Linear(in_features=self.input_size[0], out_features=self.hidden_size),
            nn.ReLU(),
        )
        self.after_torso()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Ensure JIT components are properly handled if necessary
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize JIT components if necessary

    def after_torso(self):
        self.lstmcell = nn.LSTMCell(
            input_size=self.hidden_size, hidden_size=self.hidden_size
        )

        # value
        self.value = nn.Linear(in_features=self.hidden_size, out_features=1)

        # policy
        self.logits = nn.Linear(
            in_features=self.hidden_size, out_features=self.n_outputs
        )

    @jit.ignore
    def get_dist_info_act(self, probs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.logits, dist.log_prob(action)

    @jit.ignore
    def get_dist_info_forward(self, probs: Tensor, behav_acts: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        dist = Categorical(probs)
        return dist.logits, dist.log_prob(behav_acts), dist.entropy()

    @jit.export
    def act(self, obs: Tensor, lstm_hxs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tuple[Tensor, Tensor]]:
        with torch.no_grad():
            x = self.body.forward(obs)  # x: (feat,)
            hx, cx = self.lstmcell(x, lstm_hxs)

            logits = self.logits(hx)
            probs = F.softmax(logits, dim=-1)  # logits: (batch, feat)

            action, dist_logits, dist_log_prob = self.get_dist_info_act(probs)

        return (
            action,
            dist_logits,
            dist_log_prob,
            (hx, cx),
        )

    @jit.export
    def forward(self, obs: Tensor, lstm_hxs: Tuple[Tensor, Tensor], behaviour_acts: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, seq, *sha = self.batch_size, self.sequence_length, *self.input_size
        hx, cx = lstm_hxs

        behav_acts = behaviour_acts.long().squeeze(-1)
        
        obs = obs.contiguous().view(batch * seq, *sha)
        x = self.body.forward(obs)
        x = x.view(batch, seq, self.hidden_size)  # (batch, seq, hidden_size)

        output = []
        for i in range(seq):
            hx, cx = self.lstmcell(x[:, i], (hx, cx))
            output.append(hx)
        output = torch.stack(output, dim=1)  # (batch, seq, feat)

        value = self.value(output)  # (batch, seq, 1)
        logits = self.logits(output)
        probs = F.softmax(logits, dim=-1)  # logits: (batch, feat)
        
        dist_logits, dist_log_prob, dist_entropy = self.get_dist_info_forward(probs, behav_acts)

        return (
            dist_logits.view(batch, seq, -1),
            dist_log_prob.view(batch, seq, -1),
            dist_entropy.view(batch, seq, -1),
            value.view(batch, seq, -1),
        )