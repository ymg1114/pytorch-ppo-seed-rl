import uuid
import time
import grpc

import numpy as np
import torch

from env.base import EnvBase

from buffers import grpc_service_pb2 as Pb2
from buffers import grpc_service_pb2_grpc as Pb2gRPC

from utils.utils import (
    serialize,
    deserialize,
    SERVER_IP,
    SERVER_PORT,
)


class WorkerRPC:
    def __init__(
        self,
        args,
        worker_name,
    ):
        self.args = args
        self.device = args.device  # cpu
        self.env = EnvBase(args)

        self.worker_name = worker_name

        # gRPC 채널 생성 (서버의 주소와 포트로 연결)
        channel = grpc.insecure_channel(f"{SERVER_IP}:{SERVER_PORT}")
        self.stub = Pb2gRPC.RunnerServiceStub(channel)

    def gene_rollout(self):
        print("Build Environment for {}".format(self.worker_name))

        self.num_epi = 0
        while True:
            # 초기화
            obs = self.env.reset()
            id_ = f"{uuid.uuid4()}-{self.worker_name}"  # 고유한 난수 + 워커 이름
            lstm_hx = (
                torch.zeros(self.args.hidden_size),
                torch.zeros(self.args.hidden_size),
            )  # (h_s, c_s) / (hidden,)
            self.epi_rew = 0
            is_fir = True  # first frame

            for _ in range(self.args.time_horizon):
                # 데이터를 직렬화하여 ActRequest 메시지 생성
                act_request = Pb2.ActRequest(
                    obs=serialize(obs),
                    lstm_hx=serialize(lstm_hx),
                    id=id_,
                )
                # Act 메서드를 통해 서버에 요청
                act_response = self.stub.Act(act_request)

                # 서버 응답을 역직렬화하여 사용
                act = deserialize(act_response.act)
                logits = deserialize(act_response.logits)
                log_prob = deserialize(act_response.log_prob)
                lstm_hx = deserialize(act_response.lstm_hx)

                obs, rew, done = self.env.step(act.item())
                self.epi_rew += rew

                report_data_request = Pb2.ReportRequest(
                    data_request=Pb2.ReportDataRequest(
                        rew=rew * self.args.reward_scale,
                        is_fir=bool(is_fir),
                        done=bool(done),
                        id=id_,
                    )
                )
                # Report 메서드를 통해 서버에 요청
                _ = self.stub.Report(report_data_request)

                is_fir = False

                time.sleep(0.05)

                if done:
                    break

            report_stat_request = Pb2.ReportRequest(
                stat_request=Pb2.ReportStatRequest(epi_rew=self.epi_rew, id=id_)
            )

            # Report 메서드를 통해 서버에 요청
            _ = self.stub.Report(report_stat_request)

            self.epi_rew = 0
            self.num_epi += 1
