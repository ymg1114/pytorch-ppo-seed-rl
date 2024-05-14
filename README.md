# pytorch-ppo-seed-rl


## Using torch.distributed.rpc for Remote Procedure Calls (RPC)
Note: This package is deprecated and no longer supported.

This implementation serves as a reference for the overall program architecture.

For a nice and practical solution, consider using **gRPC.**

## Caution!
High CPU usage limits the scalability of worker nodes.

Learning Environmen is limited to the discrete action space of `CartPole-v1`.

## How to run
`python main.py run_master`

`python main.py run_slave`
