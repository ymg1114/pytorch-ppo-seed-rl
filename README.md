# pytorch-ppo-seed-rl


## Brief Overview
This implementation uses **gRPC** to support the **SEED RL** architecture.

## Generate gRPC Stub Code
To generate the gRPC stub code from the `.proto` file, you can use the following command:

`python -m grpc_tools.protoc -I=buffers --python_out=buffers --grpc_python_out=buffers buffers/grpc_service.proto`.

And also need to modify a portion of the code in the `buffer/grpc_service_pb2_grpc.py` script as shown below.

![image](https://github.com/user-attachments/assets/c1e8c10c-00dd-4944-b4cb-0b2eb1651989)


## Caution!
High CPU usage limits the scalability of worker nodes.

Learning Environment is limited to the discrete action space of `CartPole-v1`.

## How to run
`python main.py run_server`

`python main.py run_client`
