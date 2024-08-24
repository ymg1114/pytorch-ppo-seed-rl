# pytorch-ppo-seed-rl


## Brief Overview
This implementation uses **gRPC** to support the **SEED RL** architecture.

## Generate gRPC Stub Code
To generate the gRPC stub code from the `.proto` file, you can use the following command

`python -m grpc_tools.protoc -I=buffers --python_out=buffers --grpc_python_out=buffers buffers/grpc_service.proto`

## Caution!
High CPU usage limits the scalability of worker nodes.

Learning Environment is limited to the discrete action space of `CartPole-v1`.

## How to run
`python main.py run_server`

`python main.py run_client`
