import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical, Normal


class MlpLSTMBase(nn.Module):
    def __init__(self, f, n_outputs, sequence_length, hidden_size):
        super().__init__()
        self.input_size = f
        self.n_outputs = n_outputs
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size

        self.body = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=self.hidden_size),
            nn.ReLU(),
        )
        self.after_torso()

        self.CT = Categorical

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

    def act(self, obs, lstm_hxs):
        with torch.no_grad():
            x = self.body.forward(obs)  # x: (feat,)
            hx, cx = self.lstmcell(x, lstm_hxs)

            dist = self.get_dist(hx)
            action = dist.sample().detach()

            # TODO: 좀 이상한 코드..
            logits = (
                dist.logits.detach()
                if hasattr(dist, "logits")
                else torch.zeros(action.shape)
            )
        return (
            action,
            logits,
            dist.log_prob(action).detach(),
            (hx.detach(), cx.detach()),
        )

    def get_dist(self, x):
        logits = self.logits(x)
        probs = F.softmax(logits, dim=-1)  # logits: (batch, feat)
        return self.CT(probs)

    def forward(self, obs, lstm_hxs, behaviour_acts):
        batch, seq, *sha = obs.size()
        hx, cx = lstm_hxs

        obs = obs.contiguous().view(batch * seq, *sha)
        x = self.body.forward(obs)
        x = x.view(batch, seq, self.hidden_size)  # (batch, seq, hidden_size)

        output = []
        for i in range(seq):
            hx, cx = self.lstmcell(x[:, i], (hx, cx))
            output.append(hx)
        output = torch.stack(output, dim=1)  # (batch, seq, feat)

        value = self.value(output)  # (batch, seq, 1)
        dist = self.get_dist(output)  # (batch, seq, num_acts)

        if isinstance(dist, Categorical):
            behav_acts = behaviour_acts.squeeze(-1)
        else:
            assert isinstance(dist, Normal)
            behav_acts = behaviour_acts

        log_probs = dist.log_prob(behav_acts)
        entropy = dist.entropy()  # (batch, seq)

        # TODO: 좀 이상한 코드..
        logits = (
            dist.logits.view(batch, seq, -1)
            if hasattr(dist, "logits")
            else torch.zeros(behaviour_acts.shape)
        )
        return (
            logits,
            log_probs.view(batch, seq, -1),
            entropy.view(batch, seq, -1),
            value.view(batch, seq, -1),
        )