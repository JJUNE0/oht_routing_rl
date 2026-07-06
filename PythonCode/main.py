import sys
# Windows cp949 콘솔에서 한글/이모지 print 시 UnicodeEncodeError 방지
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
except Exception:
    pass

import PClient
import argparse
import socket
import time
import traceback
from pathlib import Path
import json
from datetime import datetime
import os

HOST = '127.0.0.1'
PORT = 9100
# 시뮬이 보내는 정상 명령 코드 집합. 이 밖의 v = TCP 스트림 desync 신호 →
# 가비지를 파싱(→ hang/KeyError)하지 말고 소켓을 닫고 재접속해 회복.
EXPECTED_SIMULATION_STATES = {0, 1, 2, 3, 4, 5, 6}


def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region",
        action="store_true",
        help="Run ClientAlgorithm_region region-token TD7 harness.",
    )
    return parser.parse_args()


def CreateClient(use_region):
    if use_region:
        import ClientAlgorithm_region as ClientAlgorithmModule

        print("[main] using ClientAlgorithm_region (--region)")
    else:
        import ClientAlgorithm as ClientAlgorithmModule

        print("[main] using ClientAlgorithm")
    return ClientAlgorithmModule.ClientAlgorithm()


def ReadConfig(path='wpconfig.json'):
    base = Path(__file__).resolve().parent.parent.parent
    cfg_file = base / path
    if os.path.isfile(cfg_file):
        with cfg_file.open() as f:
            data = json.load(f)
            return data['MyProcessListenPort']
    return PORT;


def SendSimDataOnly(pclient, client):
    pclient.RecieveSimulationSnapshotData()
    client.UpdateDatas(pclient)

def SendAndReceiveRailLineCost(pclient, client):
    pclient.RecieveSimulationActiveData()
    client.Algorithm(pclient)
    pclient.SendRailLineCostMessage()
    client.AlgorithmAfter(pclient)


def ReceiveAndSendSingleOHTData(pclient, client):
    key, curLine, destination = pclient.RecieveSingleOHTData()
    client.UpdateOHTRoute(pclient,key, curLine, destination, pclient.OHT_DIC[key].RouteList)
    pclient.SendSingleRoute(key)

def ReceiveRerouteData(pclient, client):
    need_oht_list,from_nodes, to_nodes = pclient.GetNeedReRouteOht()
    reroute_dic = client.ReRoute(pclient, need_oht_list,from_nodes,to_nodes)
    pclient.SendReRoute(reroute_dic)
    pass

def AssignOHT(pclient, client):
    pclient.RecieveSimulationSnapshotData()
    job_list = pclient.GetAssignCommand()
    oht_dict = client.Assign(pclient, job_list)
    pclient.SendAssignOht(oht_dict)
    pass


def main():
    args = ParseArgs()
    PORT = ReadConfig()
    is_connect = False

    # ClientAlgorithm을 bind/listen 이전에 생성한다.
    # PClient.__init__은 핸드셰이크 전체를 동기로 완료하므로, 그 직후 시뮬은 즉시
    # 루프를 시작한다. CA 생성이 핸드셰이크 뒤에 오면 7초간 시뮬 데이터를 못 받아
    # 소켓 버퍼 오버플로 / 시뮬 타임아웃이 발생한다.
    # 포트를 열기 전에 생성하면 시뮬이 CA 준비 전에 연결 자체를 못 하므로 안전하다.
    client = CreateClient(args.region)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST,PORT));
    server_socket.listen(1);
    sockets =[];
    while True:
        try :
            if is_connect == False:
                print(datetime.now().strftime('%Y.%m.%d - %H:%M:%S'))
                print("Socket 서버 응답 대기 ")
                s, client_address  = server_socket.accept()
                with s:
                    sockets.append(s);
                    pclient = PClient.PClient(s)
                    client.on_new_connection()
                    is_connect = True
                    print(datetime.now().strftime('%Y.%m.%d - %H:%M:%S'))
                    print("Socket 서버와 연결되었습니다.")
                    print(HOST)
                    print(PORT)
                    pclient.WriteAdminLog("Socket 서버와 연결되었습니다. ")
                    try :
                        while True:
                            print(datetime.now().strftime('%Y.%m.%d - %H:%M:%S'))
                            print("RecieveSimulationStandardData.")
                            pclient.WriteAdminLog("RecieveSimulationStandardData.")
                            v = pclient.RecieveSimulationStandardData()
                            print("v: ", v)
                            # 예상밖 v = 스트림 desync 신호. 가비지 파싱(→ hang/KeyError) 대신 소켓 닫고
                            # 재접속해 회복. client는 루프 밖 생성이라 재접속에도
                            # 리플레이 버퍼/네트워크 유지됨 → 학습 연속.
                            if v not in EXPECTED_SIMULATION_STATES:
                                pending = pclient.PeekPending(64)
                                hexs = ' '.join(f'{b:02X}' for b in pending[:32])
                                message = (f"[DESYNC] 예상밖 v={v} (0x{v:02X}) — 스트림 정렬 깨짐. "
                                           f"대기 {len(pending)}B (세션누적 {pclient.TOTAL_BYTES_READ}B): {hexs}. "
                                           f"소켓 닫고 재접속.")
                                print(message)
                                try:
                                    pclient.WriteAdminLog(message)
                                except Exception:
                                    pass
                                raise RuntimeError(message)
                            if v == 0:
                                pclient.WriteAdminLog("SendAndReceiveRailLineCost.")
                                SendAndReceiveRailLineCost(pclient, client)
                            elif v == 2:
                                client.Reset(pclient)
                            elif v == 3:
                                SendSimDataOnly(pclient, client)
                            elif v == 4:
                                ReceiveAndSendSingleOHTData(pclient, client)
                            elif v == 5:
                                ReceiveRerouteData(pclient, client)
                            elif v == 6:
                                AssignOHT(pclient, client)
                            elif v == 1:
                                # 에피소드 종료 신호(RecieveSimulationStandardData가 이미 episode_index++). 루프 계속.
                                pclient.WriteAdminLog("Simulation end signal (v=1).")

                    except IndexError as ide:
                        pclient.WriteAdminLog("IndexError: The index is out of range. Please check if it has ended normally.");
                        print("The index is out of range. Please check if it has ended normally.")
                        traceback.print_exc()
                        is_connect = False
                        s.close();
                    except ConnectionResetError as cre:
                        pclient.WriteAdminLog("ConnectionResetError: The connection was forcibly terminated by the remote host.");
                        print("ConnectionResetError: The connection was forcibly terminated by the remote host.")
                        traceback.print_exc()
                        is_connect = False
                        s.close();
                    except Exception as ex:
                        pclient.WriteAdminLog("Exception: The connection to the Socket server has been terminated.");
                        print("Exception: The connection to the Socket server has been terminated.")
                        traceback.print_exc()
                        is_connect = False
                        s.close();

        except Exception as ex:
            print(datetime.now().strftime('%Y.%m.%d - %H:%M:%S'))
            print("while Exception.")
            traceback.print_exc()   # 바깥 예외(생성/루프 등)를 숨기지 않고 출력
            is_connect = False
            time.sleep(1)

if __name__ == "__main__":
    main()
