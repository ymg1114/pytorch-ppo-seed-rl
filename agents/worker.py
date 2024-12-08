import uuid
import asyncio
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

        # gRPC 비동기 채널 생성
        # grpc.aio 클라이언트(self.channel, self.stub)는 내부적으로 현재 활성화된 이벤트 루프를 사용
        # self.loop를 명시적으로 참조하지 않아도 grpc.aio는 asyncio.get_event_loop()를 통해 현재 이벤트 루프를 가져옴
        self.loop = asyncio.get_event_loop()  # 공통 이벤트 루프 사용

    async def setup_channel(self):
        self.channel = grpc.aio.insecure_channel(f"{SERVER_IP}:{SERVER_PORT}")
        self.stub = Pb2gRPC.RunnerServiceStub(self.channel)

    async def close(self):
        if self.channel:
            await self.channel.close()

    # async def check_server_ready(self):
    #     """서버 연결 상태 확인"""

    #     try:
    #         await grpc.channel_ready_future(self.channel).result(timeout=5)
    #         print(f"[{self.worker_name}] gRPC server is ready.")
    #     except grpc.FutureTimeoutError:
    #         print(f"[{self.worker_name}] Failed to connect to gRPC server.")
    #         await self.close()
    #         raise ConnectionError("gRPC server is not ready.")

    async def safe_call(self, method, request):
        """안정적인 gRPC 호출 (재시도 포함)"""

        try:
            return await method(request)
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                print(f"[{self.worker_name}] gRPC Error: {e.code()} - {e.details()}")
                # 재시도 로직 추가
                await asyncio.sleep(1)
                return await self.safe_call(method, request)
            else:
                raise e

    async def gene_rollout(self):
        """클라이언트 사이드 메인 작업 루프"""
        
        try:
            # await self.check_server_ready()  # 서버 준비 상태 확인
            await self.setup_channel()  # gRPC 채널 초기화
            print(f"[{self.worker_name}] Starting rollout generation.")

            self.num_epi = 0
            while True:
                # 초기화
                obs = self.env.reset()
                client_id = f"{uuid.uuid4()}-{self.worker_name}"  # 고유한 난수 + 워커 이름
                lstm_hx = (
                    torch.zeros(self.args.hidden_size),
                    torch.zeros(self.args.hidden_size),
                )  # (h_s, c_s) / (hidden,)
                self.epi_rew = 0
                is_fir = True  # 첫 프레임

                for _ in range(self.args.time_horizon):
                    # 데이터를 직렬화하여 ActRequest 메시지 생성
                    act_request = Pb2.ActRequest(
                        obs=serialize(obs),
                        lstm_hx=serialize(lstm_hx),
                        id=client_id,
                    )

                    # 안정적인 Act 호출
                    act_response = await self.safe_call(self.stub.Act, act_request)

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
                            id=client_id,
                        )
                    )

                    # 안정한 Report 호출
                    await self.safe_call(self.stub.Report, report_data_request)

                    is_fir = False
                    await asyncio.sleep(0.1)

                    if done:
                        break

                report_stat_request = Pb2.ReportRequest(
                    stat_request=Pb2.ReportStatRequest(epi_rew=self.epi_rew, id=client_id)
                )

                # 안정한 Report 호출
                await self.safe_call(self.stub.Report, report_stat_request)

                self.epi_rew = 0
                self.num_epi += 1

                print(f"[{self.worker_name}] Episode {self.num_epi} complete.")

        except grpc.aio.AioRpcError as e:
            print(f"[{self.worker_name}] gRPC Error: {e.code()} - {e.details()}")
        except ConnectionError as e:
            print(f"[{self.worker_name}] {e}")
        except Exception as e:
            print(f"[{self.worker_name}] Unexpected error: {e}")
        finally:
            await self.close()
