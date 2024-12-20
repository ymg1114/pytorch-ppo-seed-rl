import os
import asyncio

# import time
import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

from utils.utils import ExecutionTimer

from ..compute_loss import compute_gae


async def learning(parent, timer: ExecutionTimer):
    assert hasattr(parent, "batch_queue")
    parent.idx = 0

    while not parent.stop_event.is_set():
        batch_args = None
        with timer.timer("learner-throughput", check_throughput=True):
            with timer.timer("learner-batching-time"):
                batch_args = await parent.batch_queue.get()

            if batch_args is not None:
                with timer.timer("learner-forward-time"):
                    # Basically, mini-batch-learning (batch, seq, feat)
                    obs, act, rew, _, behav_log_probs, is_fir, hx, cx = parent.to_gpu(
                        *batch_args
                    )

                    # epoch-learning
                    for _ in range(parent.args.K_epoch):
                        lstm_states = (
                            hx[:, 0],
                            cx[:, 0],
                        )  # (batch, seq, hidden) -> (batch, hidden)

                        # on-line model forwarding
                        logits, log_probs, entropy, value = parent.model(
                            obs,  # (batch, seq, *sha)
                            lstm_states,  # ((batch, hidden), (batch, hidden))
                            act,  # (batch, seq, 1)
                        )
                        with torch.no_grad():
                            td_target = (
                                rew[:, :-1]
                                + parent.args.gamma * (1 - is_fir[:, 1:]) * value[:, 1:]
                            )
                            delta = td_target - value[:, :-1]

                            gae = compute_gae(
                                delta, parent.args.gamma, parent.args.lmbda
                            )  # ppo-gae (advantage)

                        ratio = torch.exp(
                            log_probs[:, :-1] - behav_log_probs[:, :-1]
                        )  # a/b == exp(log(a)-log(b))

                        surr1 = ratio * gae
                        surr2 = (
                            torch.clamp(
                                ratio,
                                1 - parent.args.eps_clip,
                                1 + parent.args.eps_clip,
                            )
                            * gae
                        )

                        loss_policy = -torch.min(surr1, surr2).mean()
                        loss_value = F.smooth_l1_loss(value[:, :-1], td_target).mean()
                        policy_entropy = entropy[:, :-1].mean()

                        loss = (
                            parent.args.policy_loss_coef * loss_policy
                            + parent.args.value_loss_coef * loss_value
                            - parent.args.entropy_coef * policy_entropy
                        )

                        detached_losses = {
                            "policy-loss": loss_policy.detach().cpu(),
                            "value-loss": loss_value.detach().cpu(),
                            "policy-entropy": policy_entropy.detach().cpu(),
                            "ratio": ratio.detach().cpu(),
                        }

                        with timer.timer("learner-backward-time"):
                            parent.optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                parent.model.parameters(),
                                parent.args.max_grad_norm,
                            )
                            print(
                                "loss: {:.5f} original_value_loss: {:.5f} original_policy_loss: {:.5f} original_policy_entropy: {:.5f} ratio-avg: {:.5f}".format(
                                    loss.item(),
                                    detached_losses["value-loss"],
                                    detached_losses["policy-loss"],
                                    detached_losses["policy-entropy"],
                                    detached_losses["ratio"].mean(),
                                )
                            )
                            parent.optimizer.step()

                # 인퍼런스 전용 모델 업데이트
                parent.infer_model.load_state_dict(parent.model.state_dict())

                if parent.idx % parent.args.loss_log_interval == 0:
                    await parent.log_loss_tensorboard(timer, loss, detached_losses)

                if parent.idx % parent.args.model_save_interval == 0:
                    torch.save(
                        parent.model.state_dict(),
                        os.path.join(
                            parent.args.model_dir, f"{parent.args.algo}_{parent.idx}.pt"
                        ),
                    )

                parent.idx += 1

        await asyncio.sleep(0.001)
