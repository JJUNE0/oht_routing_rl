import socket
from datetime import datetime
import math
import sys
import time
import os, sys
from RailLineCost import RailLineCost
from Job import Job
from Oht import Oht
from RailLine import RailLine
from LinePassTime import LinePassTime
from ohtLinePassTime import ohtLinePassTime
from ohtPos import ohtPos
from OHTCommandTime import OHTCommandTime
from collections import OrderedDict
class PClient:
       BUFFER_SIZE= int(8912);
       RAILINE_COUNT = 0;
       OHT_COUNT =0;
       RECV_TIMEOUT = 60      # recv 무응답 감지 주기(초). 초과해도 연결은 유지하고 로그만 남긴다(진단용).
       LAST_V = -1            # 마지막으로 읽은 명령 바이트(hang 진단 로그용)
       TOTAL_BYTES_READ = 0   # 세션 누적 수신 바이트(스트림 위치 추적용)
       MESSAGE_FILE_PATH = "C:\\PINOKIO\\PYTHON_MESSAGE.txt";
       communcation_type = -1

       RAILLINECOST_DIC = {};
       RAILINE_SIM_ID_DIC = {};
       RAILLINE_DIC= {};

       CompletedCommandCount = 0 ;
       TransferrCommandCount = 0 ;
       WaitingCommandCount = 0;
       QueuedCommandCount = 0 ;

       ReRoueDic = {}
       OHT_DIC = {};
       JOB_DIC = {};
       IsFaillMessage = False;

       FILE_NAME ='';
       SimStartTimeSec = 0;
       SimEndTimeSec = 0 ;
       IsSendOnly = False

       TotalTat= 0;
       TotalOhtOperationRate = 0 ;
       SimTime = 0;
    #    file_path = "secData.txt"
       file_path = ".txt"
       file_path_TCP = ".txt"
       file_extension = ".txt"
       fileAdmin_path = "adminData.txt"
       secData ={}

       episode_index = 0
       def __init__(self, socket, sim_end_time=45000):
           self.client_socket = socket
           self.requested_end_time = int(sim_end_time)
           if self.requested_end_time <= 0:
               raise ValueError("sim_end_time must be positive")
           try:
               self.client_socket.settimeout(self.RECV_TIMEOUT)  # 무한 hang 진단용; 타임아웃시 RecieveMessage가 로그 후 계속 대기
           except Exception:
               pass
           self.SendConnectMessage(2);
           self.RecieveInitializeMessage();
           self.RecieveInitializeMessage2();
           self.SendStartPythonStartTime(0);
           self.SendDijkstraUpdateTime(1);
           self.SendReroutingUpdateTime(7);
           # This raw fixed-width field must occur exactly once here.
           self.SendEndTime(self.requested_end_time);
           self.RecieveSetPara();
           self.file_path = datetime.now().strftime('%Y%m%d%H%M%S')  + self.file_path;
           self.file_path_TCP = datetime.now().strftime('%Y%m%d%H%M%S')  + self.file_path_TCP;

       def RecieveSimulationStandardData(self):
         isEnd = self.RecieveEndSim();
         if isEnd == 2:
           self.JOB_DIC = {};
           self.RecieveInitializeMessage();
           self.RecieveInitializeMessage2();
           self.SendStartPythonStartTime(0);
           self.SendDijkstraUpdateTime(1);
           self.SendReroutingUpdateTime(7);
           # The reset handshake also owns exactly one end-time field.
           self.SendEndTime(self.requested_end_time);
           self.RecieveSetPara();
           return 2;
           
         if isEnd == 1:
           self.episode_index += 1
           return 1;

         return isEnd;
    
       def RecieveSimulationActiveData(self):

          self.SendDijkstraUpdateTime(1);
          self.SendReroutingUpdateTime(7);

          self.secData ={};

          tstart = time.time()

          start = time.time()
          self.RecieveRailLineData();
          end = time.time()
          self.secData["RecieveRailLineData"] = end - start;

          start = time.time()
          self.RecieveJobData();
          end = time.time()
          self.secData["RecieveJobData"] = end - start;

          start = time.time()
          self.RecieveOHTData();
          end = time.time()
          self.secData["RecieveOHTData"] = end - start;
         # self.RecieveBumpingOHT();
          tend = time.time()
          self.secData["RecieveSimulationActiveData"] = tend - tstart;

       def WriteLog(self):
           folder =os.path.join(os.path.dirname(__file__), 'log')
           if (os.path.isdir(folder) == False):
                os.mkdir(folder)
           filePath =os.path.join(folder, self.file_path)
           if ( os.path.exists(filePath)):
                file_size_bytes  = os.path.getsize(filePath)
                if (file_size_bytes  > 1024 * 1024* 10):
                    self.file_path =datetime.now().strftime('%Y%m%d%H%M%S')  + self.file_extension;
                    filePath =os.path.join(folder, self.file_path)
           with open(filePath, "a") as file:
               file.write(datetime.now().strftime('%Y.%m.%d - %H:%M:%S') )
               for item in self.secData:
                    file.write("," + item + "/" + str( round( self.secData[item],3)) )
               file.write("\n")
       def WriteTCPLog(self, line):
           folder =os.path.join(os.path.dirname(__file__), 'log')
           if (os.path.isdir(folder) == False):
                os.mkdir(folder)
           filePath =os.path.join(folder, self.file_path_TCP)
           if ( os.path.exists(filePath)):
                file_size_bytes  = os.path.getsize(filePath)
                if (file_size_bytes  > 1024 * 1024* 10):
                    self.file_path_TCP =datetime.now().strftime('%Y%m%d%H%M%S')  + self.file_extension;
                    filePath =os.path.join(folder, self.file_path_TCP)
           with open(filePath, "a") as file:
               file.write(datetime.now().strftime('%Y.%m.%d - %H:%M:%S') )
               file.write(line )
               file.write("\n")
       def WriteAdminLog(self, line):
           folder =os.path.join(os.path.dirname(__file__), 'log')
           if (os.path.isdir(folder) == False):
                os.mkdir(folder)
           filePath =os.path.join(folder, self.fileAdmin_path)
           with open(filePath, "a") as file:
               file.write(datetime.now().strftime('%Y.%m.%d - %H:%M:%S') )
               file.write(line )
               file.write("\n")
       def RecieveSimulationSnapshotData(self):
          self.RecieveRailLineData();
          # self.RecieveJobData();  # 새 시뮬은 스냅샷에 Job 데이터 안 보냄 (PythonCode_origin 참조)
          self.RecieveOHTData();

       def SendIsEnd(self, isEnd):
           byteArray =  bytearray(1);
           byteArray[0] = isEnd
           self.SendMessage(byteArray);
         #  self.RecieveDynamicPassTimePara();  # 새 시뮬은 이 패킷을 보내지 않음 (PythonCode_origin 참조)

       def SendDijkstraUpdateTime(self, updateTimeSec):
           iupdateTimeSec = round(updateTimeSec * 10);
           byteArray =  bytearray(2);
           self.SetByteHexa_2Legnth(int(iupdateTimeSec), byteArray , 0);
           self.SendMessage(byteArray);

       def SendReroutingUpdateTime(self, updateTimeSec):
           iupdateTimeSec = round(updateTimeSec * 10);
           byteArray =  bytearray(2);
           self.SetByteHexa_2Legnth(int(iupdateTimeSec), byteArray , 0);
           self.SendMessage(byteArray);

       def SendStartPythonStartTime(self, updateTimeSec):
           iupdateTimeSec = round(updateTimeSec );
           byteArray =  bytearray(3);
           self.SetByteHexa_3Legnth(int(iupdateTimeSec), byteArray , 0);
           self.SendMessage(byteArray);

       def SendEndTime(self, updateTimeSec):
           iupdateTimeSec = round(updateTimeSec );
           byteArray =  bytearray(3);
           self.SetByteHexa_3Legnth(int(iupdateTimeSec), byteArray , 0);
           self.SendMessage(byteArray);
           
       def GetSimIDByTCPID(self, currentID):
           if  currentID in self.RAILLINE_DIC:
               return self.RAILLINE_DIC[currentID].ID;
           else:
               return -1;

       def GetTCPIDBySimID(self, simID):
           if  simID in self.RAILINE_SIM_ID_DIC:
               return self.RAILINE_SIM_ID_DIC[simID];
           else:
               return -1;
           
       def RecieveSetPara(self):
           recieveMessage = self.RecieveMessage(16);

       def RecieveDynamicPassTimePara(self):
           raillineCount = 0 ;
           recieveMessage = self.RecieveMessage(self.BUFFER_SIZE);
           railLineIndex = 0;
           while (raillineCount < self.RAILINE_COUNT):
               if (railLineIndex + 9 < self.BUFFER_SIZE):
                   simID = self.GetBase10Value_2(recieveMessage, railLineIndex);
                   railLineIndex += 2;

                   if simID == 0:
                       railLineIndex = self.BUFFER_SIZE;
                       continue;

                   id = self.GetBase10Value_2(recieveMessage, railLineIndex);
                   railLineIndex += 2

                 #  self.RAILLINE_DIC[id].PossibleVHLCount = recieveMessage[railLineIndex]
                   railLineIndex += 1

              #     self.RAILLINE_DIC[id].DynamicWeight =   float(self.GetBase10Value_2(recieveMessage, railLineIndex) - 1000) / 100 ;
                   railLineIndex += 2

                   passTimeCount =  self.GetBase10Value_2(recieveMessage, railLineIndex);
                   railLineIndex += 2;

            #      self.RAILLINE_DIC[id].DynamicWeights = [];
                   ptc = 0;
                   while (ptc < passTimeCount):
                       
                       beforeDynamicWeight =float( self.GetBase10Value_2(recieveMessage, railLineIndex) - 1000) /100;
                       railLineIndex += 2;
                       currentDynamicWeight =float( self.GetBase10Value_2(recieveMessage, railLineIndex)- 1000) /100;
                       railLineIndex += 2;
                       passWeight =float( self.GetBase10Value_2(recieveMessage, railLineIndex)- 1000) /100;
                       railLineIndex += 2;
                       passTime =float( self.GetBase10Value_2(recieveMessage, railLineIndex)) /100;
                       railLineIndex += 2;
                       ptc += 1;
            
                   raillineCount +=1;
               else :
                   recieveMessage = self.RecieveMessage(self.BUFFER_SIZE);
                   railLineIndex = 0;


       def RecieveEndSim(self):
           recieveMessage = self.RecieveMessage(1);
           self.LAST_V = recieveMessage[0];
           return recieveMessage[0];

       def RecieveRailLineData(self):
           raillineCount = 0

           start = time.time()
           recieveMessage = self.RecieveMessage(self.BUFFER_SIZE)
           end = time.time()
           if "RAIL_TCP_TIME" in self.secData:
            self.secData["RAIL_TCP_TIME"] += (end - start)
           else:
            self.secData["RAIL_TCP_TIME"] = 0
           railLineIndex = 0
           while raillineCount < self.RAILINE_COUNT:
               if railLineIndex < self.BUFFER_SIZE:
                   # 여기부터
                   start = time.time()
                   id = self.GetBase10Value_2(recieveMessage, railLineIndex)
                   railLineIndex += 2
                   if id == 0:
                       railLineIndex = self.BUFFER_SIZE
                       continue
                   if id not in self.RAILLINE_DIC.keys():
                       # DESYNC: 0이 아닌데 알 수 없는 라인 id = 스트림 정렬 깨짐. 가비지 파싱 말고
                       # 예외 → main의 except가 소켓 닫고 재접속(동료 코드처럼 회복).
                       dmsg = ("[DESYNC] RailLineData id=" + str(id) + "(알수없음) @idx="
                               + str(railLineIndex - 2) + " raillineCount=" + str(raillineCount)
                               + "/" + str(self.RAILINE_COUNT) + " 세션누적=" + str(self.TOTAL_BYTES_READ) + "B")
                       print(dmsg)
                       try:
                           self.WriteAdminLog(dmsg)
                       except Exception:
                           pass
                       raise RuntimeError(dmsg)
                   sim_avg_speed = self.GetBase10Value_2(recieveMessage, railLineIndex) / 10
                   self.RAILLINE_DIC[id].SimAvgSpeed = sim_avg_speed
                   railLineIndex += 2
                   self.RAILLINE_DIC[id].PredictedOHTIDList = []
                   predictPacssOhtCount = recieveMessage[railLineIndex]
                   railLineIndex += 1
                   self.RAILLINE_DIC[id].PredictedOHTCount = predictPacssOhtCount
                   predictCount = 0
                   end = time.time()
                   if "RAILLINE_DIC" in self.secData:
                       self.secData["RAILLINE_DIC"] += (end - start)
                   else:
                       self.secData["RAILLINE_DIC"] = 0
                    #
                   
                   start = time.time()

                   while (predictCount < predictPacssOhtCount):
                       ohtID = self.GetBase10Value_2(recieveMessage, railLineIndex)
                       self.RAILLINE_DIC[id].PredictedOHTIDList.append(ohtID)
                       railLineIndex += 2
                       predictCount += 1

                   if (len(self.RAILLINE_DIC[id].PredictedOHTIDList) != predictPacssOhtCount):
                       d = 0
                    
                   end = time.time()
                   if   "predictPacssOhtCount" in self.secData:
                       self.secData["predictPacssOhtCount"] +=  (end - start);
                   else:
                       self.secData["predictPacssOhtCount"] = 0;
                   
                   # 여기부터
                   start = time.time()
                   
                   self.RAILLINE_DIC[id].OhtList = []
                   ohtListCount = recieveMessage[railLineIndex]
                   railLineIndex += 1
                   ohtCount = 0
                   self.RAILLINE_DIC[id].IdleOHTCount = 0
                   self.RAILLINE_DIC[id].ReservationPortCount = 0
                   
                   end = time.time()
                   
                   if "RAIL_OHT_ListCount" in self.secData:
                       self.secData["RAIL_OHT_ListCount"] += (end - start)
                   else:
                       self.secData["RAIL_OHT_ListCount"] = 0
                    #

                   start = time.time()
                   while (ohtCount < ohtListCount):
                       ohtID = self.GetBase10Value_2(recieveMessage, railLineIndex)
                       self.RAILLINE_DIC[id].OhtList.append(ohtID)
                       railLineIndex +=2
                       ohtCount += 1
                   end = time.time()
                   if   "ohtListCount" in self.secData:

                        self.secData["ohtListCount"] +=  (end - start);
                   else:
                       self.secData["ohtListCount"] = 0;
                   raillineCount +=1
               else :
                   start = time.time()
                   recieveMessage = self.RecieveMessage(self.BUFFER_SIZE)
                   end = time.time()
                   if "RAIL_TCP_TIME" in self.secData:
                    self.secData["RAIL_TCP_TIME"] += (end - start)
                   else:
                    self.secData["RAIL_TCP_TIME"] = 0
                   railLineIndex = 0
               




       def RecieveOHTData(self):
           ohtCount = 0

           start = time.time()
           recieveMessage = self.RecieveMessage(self.BUFFER_SIZE)
           end = time.time()
           if "OHT_TCP_TIME" in self.secData:
            self.secData["OHT_TCP_TIME"] += (end - start)
           else:
            self.secData["OHT_TCP_TIME"] = 0

           railLineIndex = 0

           while (ohtCount < self.OHT_COUNT):
               if  (railLineIndex + 23 < self.BUFFER_SIZE):
                   id = self.GetBase10Value_2(recieveMessage, railLineIndex)
                   railLineIndex += 2
                   if id == 0:
                       railLineIndex = self.BUFFER_SIZE
                       continue
                   if id not in self.OHT_DIC:
                       # DESYNC: 0이 아닌데 알 수 없는 OHT id (원본은 여기서 KeyError로 죽음). 예외 →
                       # main의 except가 소켓 닫고 재접속(동료 코드처럼 회복).
                       omsg = ("[DESYNC] OHTData id=" + str(id) + "(알수없음) @idx=" + str(railLineIndex - 2)
                               + " ohtCount=" + str(ohtCount) + "/" + str(self.OHT_COUNT)
                               + " 세션누적=" + str(self.TOTAL_BYTES_READ) + "B")
                       print(omsg)
                       try:
                           self.WriteAdminLog(omsg)
                       except Exception:
                           pass
                       raise RuntimeError(omsg)
                   ohtState = recieveMessage[railLineIndex]
                   railLineIndex += 1
                   ohtOperationRate = recieveMessage[railLineIndex]
                   railLineIndex += 1
                   curDistance = float(self.GetBase10Value_3(recieveMessage, railLineIndex)) / 10
                   railLineIndex += 3
                   stopTime = recieveMessage[railLineIndex]
                   railLineIndex += 1
                   jobID = self.GetBase10Value_3(recieveMessage, railLineIndex)
                   railLineIndex += 3
                   destinationLineID = self.GetBase10Value_2(recieveMessage, railLineIndex)
                   railLineIndex += 2
                   idleTime = float(self.GetBase10Value_2(recieveMessage, railLineIndex))
                   railLineIndex += 2

                   routeCount = float(self.GetBase10Value_2(recieveMessage, railLineIndex))
                   railLineIndex += 2

                   raillineCount = 0
                   self.OHT_DIC[id].RouteList = []

                   while (raillineCount < routeCount):
                       lineID = self.GetBase10Value_2(recieveMessage, railLineIndex)
                       self.OHT_DIC[id].RouteList.append(lineID)
                       railLineIndex +=2
                       raillineCount += 1
                   passCount = float(self.GetBase10Value_2(recieveMessage, railLineIndex))
                   railLineIndex += 2
                   passLine = 0

                   ids = []

                   self.OHT_DIC[id].PassTimes = []
                   self.OHT_DIC[id].FrontOhts = {}
                   while (passLine < passCount):
                       passState = recieveMessage[railLineIndex]
                       railLineIndex += 1
                       lineID = self.GetBase10Value_2(recieveMessage, railLineIndex)
                       railLineIndex += 2
                       passTime = float(self.GetBase10Value_4(recieveMessage, railLineIndex)) / 10000
                       railLineIndex += 4
                       opt = ohtLinePassTime()
                       opt.ID = lineID
                       opt.State = passState
                       opt.PassTime = passTime
                       opt.Distance = self.RAILLINE_DIC[lineID].Distance
                       self.OHT_DIC[id].PassTimes.append(opt)
                       passLine += 1
                       frontOHTCount = recieveMessage[railLineIndex]
                       railLineIndex += 1
                       if (frontOHTCount == 0):
                           c = 0
                       self.OHT_DIC[id].FrontOhts[lineID] = []
                       for _ in range(frontOHTCount):
                          ohtId = self.GetBase10Value_2(recieveMessage, railLineIndex)
                          railLineIndex += 2
                          ohtDistance = self.GetBase10Value_3(recieveMessage, railLineIndex)
                          railLineIndex += 3
                          op = ohtPos()
                          op.ID = ohtId
                          op.Distance = float(ohtDistance) / 10
                          self.OHT_DIC[id].FrontOhts[lineID].append(op)
                   
                   if len(self.OHT_DIC[id].PassTimes) == 10:
                       check = True
                   self.OHT_DIC[id].PassDistance = 0
                   self.OHT_DIC[id].VelByLine = {}

                   if(len(self.OHT_DIC[id].NotPassTimes) > 0):
                        lpass = self.OHT_DIC[id].NotPassTimes[0]
                        idx = 0
                        for pt in self.OHT_DIC[id].PassTimes:
                            d = 0
                            t = 0
                            if idx == 0 and lpass.ID == pt.ID:
                                d = pt.Distance - lpass.Distance
                                t = pt.PassTime - lpass.PassTime
                                
                            else:
                                d = pt.Distance
                                t = pt.PassTime
                            if(d < 0):
                                d = 0
                            idx +=1
                            if t == 0:
                                continue
                            self.OHT_DIC[id].PassDistance += d
                            self.OHT_DIC[id].VelByLine[pt.ID] = d / (t)
                   else:
                       for pt in self.OHT_DIC[id].PassTimes:
                           d = pt.Distance
                           t = pt.PassTime
                           if t == 0:
                              continue
                           self.OHT_DIC[id].PassDistance += d
                           self.OHT_DIC[id].VelByLine[pt.ID] = d / (t)

                   self.OHT_DIC[id].NotPassTimes = []

                   notpassCount = recieveMessage[railLineIndex]
                   railLineIndex += 1
                   if (notpassCount > 0):
                       passState = recieveMessage[railLineIndex]
                       railLineIndex += 1
                       lineID = self.GetBase10Value_2(recieveMessage, railLineIndex)
                       railLineIndex += 2
                       passTime = float(self.GetBase10Value_4(recieveMessage, railLineIndex)) / 10000
                       opt = ohtLinePassTime()
                       opt.ID = lineID
                       opt.State = passState
                       opt.PassTime = passTime
                       opt.Distance = min(curDistance,self.RAILLINE_DIC[lineID].Distance)
                       railLineIndex += 4
                       self.OHT_DIC[id].NotPassTimes.append(opt)
                       frontOHTCount = recieveMessage[railLineIndex]
                       railLineIndex += 1
                       if (frontOHTCount == 0):
                           c = 0
                       self.OHT_DIC[id].FrontOhts[lineID] = []
                       for _ in range(frontOHTCount):
                           ohtId = self.GetBase10Value_2(recieveMessage, railLineIndex)
                           railLineIndex += 2
                           ohtDistance = self.GetBase10Value_3(recieveMessage, railLineIndex)
                           railLineIndex += 3
                           op = ohtPos()
                           op.ID = ohtId
                           op.Distance = float(ohtDistance) / 10
                           self.OHT_DIC[id].FrontOhts[lineID].append(op)

                   self.OHT_DIC[id].CmdCompleteTat = {};
                   cmdCount = self.GetBase10Value_2(recieveMessage, railLineIndex)
                   railLineIndex += 2
                   commandCount = 0;
                   while (commandCount < cmdCount):
                       cmdID = self.GetBase10Value_3(recieveMessage, railLineIndex)
                       railLineIndex += 3;
                       ohtTat = self.GetBase10Value_2(recieveMessage, railLineIndex)
                       railLineIndex += 2;
                       cmdTat = self.GetBase10Value_2(recieveMessage, railLineIndex)
                       railLineIndex += 2;

                       oct = OHTCommandTime()
                       oct.CmdID = cmdID
                       oct.OHTTat = ohtTat / 10;
                       oct.CmdTat = cmdTat / 10;

                       if cmdTat > 0:
                           
                           oct.OHTWorkTimeByCommand = oct.OHTTat / oct.CmdTat
                       else:
                           oct.OHTWorkTimeByCommand = 0.0
                       
                       
                       if  cmdID in self.OHT_DIC[id].CmdCompleteTat:
                           a=0;
                       else :
                           self.OHT_DIC[id].CmdCompleteTat = {};
                       self.OHT_DIC[id].CmdCompleteTat[cmdID] = oct
                       commandCount+=1;
 
                   ohtIndividualTatRaw = self.GetBase10Value_3(recieveMessage, railLineIndex)
                   railLineIndex += 3
                   self.OHT_DIC[id].OhtIndividualTat = ohtIndividualTatRaw / 10

                   remainingDistance = 0
                   remainingDistance = self.GetBase10Value_3(recieveMessage, railLineIndex)
                   self.OHT_DIC[id].RemainingDistanace = remainingDistance
                   railLineIndex += 3

                   remainTime = 0
                   remainTime = self.GetBase10Value_4(recieveMessage, railLineIndex)
                   railLineIndex += 4
                   if remainTime != 0:
                        remainTime = round(remainTime / 10000,4)
                   dispatchedCommand = self.GetBase10Value_3(recieveMessage, railLineIndex)
                   railLineIndex += 3
                   if (len(self.OHT_DIC[id].RouteList) > 0):
                    if (ohtState == 0):
                       self.RAILLINE_DIC[self.OHT_DIC[id].RouteList[0]].IdleOHTCount += 1
                    elif (ohtState == 2 or ohtState == 4):
                       l = len(self.OHT_DIC[id].RouteList)
                       railLineID = self.OHT_DIC[id].RouteList[l - 1]
                       self.RAILLINE_DIC[railLineID].ReservationPortCount += 1
                   self.OHT_DIC[id].OperationRate = ohtOperationRate # 2번
                   self.OHT_DIC[id].State = ohtState
                   self.OHT_DIC[id].CurrentDistance = curDistance
                   self.OHT_DIC[id].JobID = jobID
                   self.OHT_DIC[id].IdleTime = idleTime / 10
                   self.OHT_DIC[id].DestinationLine = destinationLineID
                   self.OHT_DIC[id].StopTime = stopTime
                   self.OHT_DIC[id].RemainingDistanace
                   self.OHT_DIC[id].RemainTime = remainTime
                   self.OHT_DIC[id].DispatchedCommand = dispatchedCommand
                   ohtCount += 1
               else:
                   start = time.time()
                   recieveMessage = self.RecieveMessage(self.BUFFER_SIZE)
                   end = time.time()
                   if "OHT_TCP_TIME" in self.secData:
                    self.secData["OHT_TCP_TIME"] += (end - start)
                   else:
                    self.secData["OHT_TCP_TIME"] = 0
                   railLineIndex = 0

           # === [DESYNC 원인 계측] ===========================================
           # 목적: "트레일러 off-by-one이 진짜 원인인지"를 실행으로 확정한다.
           #   (1) railLineIndex == BUFFER_SIZE-7  → 구버전 +6 코드라면 바로 이 지점에서
           #       desync했을 '트레일러 경계'다. 이게 학습 중 몇 번이나 발생하는지 카운트한다.
           #       이 카운트가 늘어나는데 (아래 +7 수정으로) 이제 안 터진다면 → 트레일러가 원인 확정.
           #   (2) 트레일러를 읽은 뒤 simTime이 비정상(역행/범위초과)이면, 스트림이 트레일러
           #       '이전'(=Job/OHT 파싱)에서 이미 어긋난 것 → 원인이 트레일러가 아니라 상류.
           #       두 신호를 분리 로깅해 원인을 단정한다.
           _trailer_boundary = (railLineIndex == self.BUFFER_SIZE - 7)
           if _trailer_boundary:
               self.TRAILER_BOUNDARY_HITS = getattr(self, "TRAILER_BOUNDARY_HITS", 0) + 1
               try:
                   self.WriteAdminLog(
                       "[CALIB] 트레일러 경계 hit railLineIndex==BUFFER_SIZE-7 "
                       "(구버전 +6이면 여기서 desync) 누적="
                       + str(self.TRAILER_BOUNDARY_HITS)
                       + " 세션누적바이트=" + str(getattr(self, "TOTAL_BYTES_READ", -1)))
               except Exception:
                   pass

           # 트레일러(opRate 1B + TAT 2B + simTime 4B = 7B)는 C#의
           # CheckBufferSizeIndex(num4, 7, bufferSize)와 정확히 같은 조건으로 버퍼를 넘겨야 한다.
           # C#: `if (num4 + 7 >= bufferSize)` 면 현재 버퍼를 flush 후 새 버퍼 0번지에 트레일러 기록.
           # 기존 코드의 +6은 off-by-one → railLineIndex가 정확히 BUFFER_SIZE-7일 때
           # C#은 flush(트레일러를 다음 버퍼로)했는데 Python은 같은 버퍼의 0패딩을 트레일러로 오독하고
           # 진짜 트레일러 버퍼를 안 읽어 다음 교환부터 스트림이 desync(가비지 v/KeyError)됐다.
           if  (railLineIndex + 7 >= self.BUFFER_SIZE):
               start = time.time()
               recieveMessage = self.RecieveMessage(self.BUFFER_SIZE)
               end = time.time()
               if "OHT_TCP_TIME" in self.secData:
                self.secData["OHT_TCP_TIME"] += (end - start)
               else:
                self.secData["OHT_TCP_TIME"] = 0
               railLineIndex = 0

           totalOhtOperationRate = recieveMessage[railLineIndex] / 100;
           railLineIndex += 1
           totalTat = self.GetBase10Value_2(recieveMessage, railLineIndex) /10

           railLineIndex += 2
           simTime = self.GetBase10Value_4(recieveMessage, railLineIndex) / 10;

           # (2) 상류 desync 탐지: 신버전 에피소드는 최대 SendEndTime(45000)s. simTime이
           # 60000s를 넘으면 4바이트 simTime 자리에서 엉뚱한 바이트를 읽은 것 = 트레일러
           # '이전'(Job/OHT 파싱)에서 이미 스트림이 어긋난 것 → 원인이 트레일러가 아닌 상류.
           # (에피소드 리셋 시 simTime이 0으로 작아지는 건 정상이라 오탐 안 되게 범위만 본다.)
           if simTime > 60000:
               self.UPSTREAM_DESYNC_HITS = getattr(self, "UPSTREAM_DESYNC_HITS", 0) + 1
               try:
                   self.WriteAdminLog(
                       "[CALIB] !!상류 desync 의심!! simTime=" + str(simTime)
                       + "(>60000s 가비지) 트레일러경계?=" + str(_trailer_boundary)
                       + " → 트레일러 이전(Job/OHT)에서 어긋남. 누적="
                       + str(self.UPSTREAM_DESYNC_HITS))
               except Exception:
                   pass

           self.TotalTat = totalTat; # 6번
           self.TotalOhtOperationRate = totalOhtOperationRate; #5번
           self.SimTime = simTime;
           ohtCount = 0;
           for v in self.OHT_DIC:
               ohtNode = self.OHT_DIC[v];
               ohtNode.OperationTAT = ohtNode.OperationRate * self.SimTime;

           c= 0;


       def RecieveBumpingOHT(self):
           bayCount = 0
           recieveMessage = self.RecieveMessage(self.BUFFER_SIZE)
           index = 0
           bayCount = self.GetBase10Value_2(recieveMessage, index)
           index += 2;
           bc= 0;
           while (bc < bayCount):
               if  (index + 2 < self.BUFFER_SIZE):
                   id = self.GetBase10Value_2(recieveMessage, index)
                   index += 2
                   if id == 0:
                       index = self.BUFFER_SIZE
                       continue
                   bumpingOhtCount = self.GetBase10Value_2(recieveMessage, index)
                   index +=2;
                   bpCount = 0;
                   while (bpCount < bumpingOhtCount):
                       ohtID = self.GetBase10Value_2(recieveMessage, index)
                       index +=2
                       bpCount += 1
                   bc+= 1;
               else:
                   recieveMessage = self.RecieveMessage(self.BUFFER_SIZE)
                   index = 0

       def RecieveSingleOHTData(self): 
           recieveMessage = self.RecieveMessage(self.BUFFER_SIZE)
           ohtIndex= 0
           ohtid = self.GetBase10Value_2(recieveMessage, ohtIndex)
           ohtIndex += 2
           ohtState = recieveMessage[ohtIndex]
           ohtIndex += 1
           self.OHT_DIC[ohtid].State = ohtState
           curLine = self.GetBase10Value_2(recieveMessage, ohtIndex)
           ohtIndex += 2
           destination = self.GetBase10Value_2(recieveMessage, ohtIndex)
           ohtIndex += 2
           
           length = self.GetBase10Value_2(recieveMessage, ohtIndex)
           ohtIndex += 2
           route_list = list()
           for _ in range(length):
               line_id = self.GetBase10Value_2(recieveMessage, ohtIndex)
               ohtIndex += 2
               route_list.append(line_id)
           
           self.OHT_DIC[ohtid].RouteList = route_list
           return ohtid,curLine,destination
                

       def RecieveJobData(self):
           cC = 0
           recieveMessage = self.RecieveMessage(self.BUFFER_SIZE)
           commandIndex = 0

           self.CompletedCommandCount = self.GetBase10Value_3(recieveMessage, commandIndex)
           commandIndex += 3

           self.TransferCommandCount = self.GetBase10Value_3(recieveMessage, commandIndex)
           commandIndex += 3
           if ( self.TransferCommandCount == 327730):
             c= 0;

           self.WaitingCommandCount = self.GetBase10Value_3(recieveMessage, commandIndex)
           commandIndex += 3
                      
           self.QueuedCommandCount = self.GetBase10Value_3(recieveMessage, commandIndex)
           commandIndex += 3

           commandCount = self.CompletedCommandCount + self.TransferCommandCount + self.WaitingCommandCount + self.QueuedCommandCount

           completedCommandCount = self.CompletedCommandCount  ;
           waitingTranferCommandCount = completedCommandCount + self.TransferCommandCount + self.WaitingCommandCount;

           start = time.time()

           while (cC < completedCommandCount ):
               if (commandIndex + 13 + 4 + 1< self.BUFFER_SIZE):
                 id = self.GetBase10Value_3(recieveMessage, commandIndex)
                 commandIndex += 3
                 if id == 0:
                     commandIndex = self.BUFFER_SIZE
                     continue
                 jobState = recieveMessage[commandIndex]

                 commandIndex+=1

                 priority = self.GetBase10Value_2(recieveMessage, commandIndex)
                 commandIndex+=2

                 isEqp = recieveMessage[commandIndex]
                 commandIndex += 1
           

                 if (id in self.JOB_DIC) == False:
                     self.JOB_DIC[id] = Job()
                     self.JOB_DIC[id].ID = id
                     self.JOB_DIC[id].Priority = priority

                 self.JOB_DIC[id].State = jobState
                 self.JOB_DIC[id].IsEqp = isEqp

                 commandIndex = self.InsertRouteInfoInJob(recieveMessage, commandIndex ,id)
                 
                 if (commandIndex == self.BUFFER_SIZE):
                    a = 0
                 


                 commandIndex = self.InsertWaitingRouteInfoInJob(recieveMessage, commandIndex,id)

                 commandIndex = self.InsertTransferRouteInfoInJob(recieveMessage, commandIndex,id)

                 fromNode = self.GetBase10Value_2(recieveMessage,commandIndex)
                 commandIndex += 2
                 toNode = self.GetBase10Value_2(recieveMessage,commandIndex)
                 commandIndex += 2

                 carrier_type_cnt = recieveMessage[commandIndex]
                 commandIndex += 1
                 self.JOB_DIC[id].CarrierTypes = []
                 for i in range(carrier_type_cnt):
                     carrier_type = recieveMessage[commandIndex]
                     commandIndex += 1
                     self.JOB_DIC[id].CarrierTypes.append(carrier_type)

                 running_area_type_cnt = recieveMessage[commandIndex]
                 commandIndex += 1
                 self.JOB_DIC[id].RunningAreaTyes = [];
                 for i in range(running_area_type_cnt):
                     running_area_type = recieveMessage[commandIndex]
                     commandIndex += 1
                     self.JOB_DIC[id].RunningAreaTyes.append(carrier_type)

                 self.JOB_DIC[id].ToNode = toNode
                 self.JOB_DIC[id].FromNode = fromNode
                 del self.JOB_DIC[id]
                 cC += 1

               else:
                   recieveMessage = self.RecieveMessage(self.BUFFER_SIZE)
                   commandIndex = 0
           end = time.time()
           self.secData["completedCommandCount"] = end - start;
           start = time.time()
           while (cC < waitingTranferCommandCount):

               if (commandIndex + 13 + 4 + 1 < self.BUFFER_SIZE):
                 id = self.GetBase10Value_3(recieveMessage, commandIndex)
                 commandIndex += 3
                 if id == 0:
                     commandIndex = self.BUFFER_SIZE
                     continue
                 jobState = recieveMessage[commandIndex]

                 commandIndex+=1

                 if(id == 23158):
                   c=0;

                 priority = self.GetBase10Value_2(recieveMessage, commandIndex)
                 commandIndex+=2

                 isEqp = recieveMessage[commandIndex]
                 commandIndex += 1
           
                 reAssignCount = recieveMessage[commandIndex]
                 commandIndex += 1

                 if (id in self.JOB_DIC) == False:
                     self.JOB_DIC[id] = Job()
                     self.JOB_DIC[id].ID = id
                     self.JOB_DIC[id].Priority = priority

 


                 self.JOB_DIC[id].State = jobState
                 self.JOB_DIC[id].IsEqp = isEqp

                 if self.JOB_DIC[id].ReAssignCount < reAssignCount:
                     self.JOB_DIC[id].Waiting_PassLines =  [];


                 self.JOB_DIC[id].ReAssignCount = reAssignCount;

                 commandIndex = self.InsertRouteInfoInJob(recieveMessage, commandIndex ,id)
                 
                 if (commandIndex == self.BUFFER_SIZE):
                    a = 0;

                 commandIndex = self.InsertWaitingRouteInfoInJob(recieveMessage, commandIndex,id)
                 commandIndex = self.InsertTransferRouteInfoInJob(recieveMessage, commandIndex,id)
                 
                 fromNode = self.GetBase10Value_2(recieveMessage,commandIndex)
                 commandIndex += 2
                 toNode = self.GetBase10Value_2(recieveMessage,commandIndex)
                 commandIndex += 2

                 self.JOB_DIC[id].ToNode = toNode
                 self.JOB_DIC[id].FromNode = fromNode
                 carrier_type_cnt = recieveMessage[commandIndex]
                 commandIndex += 1
                 self.JOB_DIC[id].CarrierTypes= []
                 for i in range(carrier_type_cnt):
                     carrier_type = recieveMessage[commandIndex]
                     commandIndex += 1
                     self.JOB_DIC[id].CarrierTypes.append(carrier_type)

                 running_area_type_cnt = recieveMessage[commandIndex]
                 commandIndex += 1
                 self.JOB_DIC[id].RunningAreaTyes = []
                 for i in range(running_area_type_cnt):
                     running_area_type = recieveMessage[commandIndex]
                     commandIndex += 1
                     self.JOB_DIC[id].RunningAreaTyes.append(carrier_type)
                 cC += 1

               else:
                   recieveMessage = self.RecieveMessage(self.BUFFER_SIZE)
                   commandIndex = 0
           end = time.time()
           self.secData["waitingTranferCommandCount"] = end - start;

           start = time.time()
           while (cC < commandCount):
               if (commandIndex + 13 + 4 + 1 < self.BUFFER_SIZE):
                 id = self.GetBase10Value_3(recieveMessage, commandIndex)
                 commandIndex += 3
                 if id == 0:
                     commandIndex = self.BUFFER_SIZE
                     continue
                 jobState = recieveMessage[commandIndex]

                 commandIndex+=1

                 priority = self.GetBase10Value_2(recieveMessage, commandIndex)
                 commandIndex+=2

                 isEqp = recieveMessage[commandIndex]
                 commandIndex += 1
           

                 if (id in self.JOB_DIC) == False:
                     self.JOB_DIC[id] = Job()
                     self.JOB_DIC[id].ID = id
                     self.JOB_DIC[id].Priority = priority

                 self.JOB_DIC[id].State = jobState
                 self.JOB_DIC[id].IsEqp = isEqp

                 commandIndex = self.InsertRouteInfoInJob(recieveMessage, commandIndex ,id)
                 
                 commandIndex = self.InsertWaitingRouteInfoInJob(recieveMessage, commandIndex,id)
                 commandIndex = self.InsertTransferRouteInfoInJob(recieveMessage, commandIndex,id)
                 fromNode = self.GetBase10Value_2(recieveMessage, commandIndex)
                 commandIndex += 2
                 toNode = self.GetBase10Value_2(recieveMessage,commandIndex)
                 commandIndex += 2

                 self.JOB_DIC[id].ToNode = toNode
                 self.JOB_DIC[id].FromNode = fromNode
                 carrier_type_cnt = recieveMessage[commandIndex]
                 commandIndex += 1
                 self.JOB_DIC[id].CarrierTypes = []
                 for i in range(carrier_type_cnt):
                     carrier_type = recieveMessage[commandIndex]
                     commandIndex += 1
                     self.JOB_DIC[id].CarrierTypes.append(carrier_type)

                 running_area_type_cnt = recieveMessage[commandIndex]
                 commandIndex += 1
                 self.JOB_DIC[id].RunningAreaTyes = []
                 for i in range(running_area_type_cnt):
                     running_area_type = recieveMessage[commandIndex]
                     commandIndex += 1
                     self.JOB_DIC[id].RunningAreaTyes.append(carrier_type)
                 cC += 1

               else:
                   recieveMessage = self.RecieveMessage(self.BUFFER_SIZE)
                   commandIndex = 0
           end = time.time()
           self.secData["commandCount"] = end - start;


       def InsertRouteInfoInJob(self, byteArr , startIndex, jobID):
            routeCount = self.GetBase10Value_2(byteArr, startIndex );
            startIndex+=2;
            rc = 0;
            self.JOB_DIC[jobID].RouteList = [];
            while (rc < routeCount):
                routeID = self.GetBase10Value_2(byteArr, startIndex);
                self.JOB_DIC[jobID].RouteList.append(routeID);
                startIndex += 2;
                rc += 1;
            return startIndex;
       
       sucess = False;
       def InsertWaitingRouteInfoInJob(self,byteArr , startIndex, jobID):
            waitingCount = self.GetBase10Value_2(byteArr, startIndex );
            startIndex += 2;
            wc = 0;

            while (wc < waitingCount):
                waitingID = self.GetBase10Value_2(byteArr, startIndex);
                waitingTime = float( self.GetBase10Value_3(byteArr, startIndex + 2)) /10;
                l = LinePassTime();
                l.ID = waitingID;
                l.PassTime =waitingTime;
                self.JOB_DIC[jobID].Waiting_PassLines.append(l);
                startIndex += 5;

                wc += 1;
            l= len(self.JOB_DIC[jobID].Waiting_PassLines);
            if (l > 20):
                del  self.JOB_DIC[jobID].Waiting_PassLines[0 : l- 20]
            return startIndex;

       def InsertTransferRouteInfoInJob(self,byteArr , startIndex, jobID):
            transferCount = self.GetBase10Value_2(byteArr, startIndex );
            startIndex+=2;
            tc = 0;
            while (tc < transferCount):
                transID = self.GetBase10Value_2(byteArr, startIndex);
                transTime =  float(self.GetBase10Value_3(byteArr, startIndex + 2))/10;
                l = LinePassTime();
                l.ID = transID;
                l.PassTime =transTime;
                self.JOB_DIC[jobID].Transfer_PassLines.append(l);
                startIndex += 5;

                tc += 1;
            l= len(self.JOB_DIC[jobID].Transfer_PassLines);
            if (l > 20):
                del  self.JOB_DIC[jobID].Transfer_PassLines[0: l- 20]
            return startIndex;

       def RecieveInitializeMessage(self):
           recieveMessage = self.RecieveMessage(39);
           self.BUFFER_SIZE = int( self.GetBase10Value_4(recieveMessage, 4));
           self.RAILINE_COUNT= int(self.GetBase10Value_3(recieveMessage, 8));
           self.OHT_COUNT= int(self.GetBase10Value_2(recieveMessage, 11));
           self.FILE_NAME = str(recieveMessage[13:32]);
           self.SimStartTimeSec =int(self.GetBase10Value_3(recieveMessage,33));
           self.SimEndTimeSec = int(self.GetBase10Value_3(recieveMessage, 36));

       def RecieveInitializeMessage2(self):
           railLineCount= 0 ;
           recieveMessage = self.RecieveMessage(self.BUFFER_SIZE);
           startIndex = 0;
           self.RAILLINE_DIC = {};
           self.RAILLINECOST_DIC = {};
           self.RAILINE_SIM_ID_DIC = {};
           while railLineCount < self.RAILINE_COUNT:
               if startIndex + 5 < self.BUFFER_SIZE:
                 id = self.GetBase10Value_2(recieveMessage, startIndex);
                 startIndex += 2;

                 simID = self.GetBase10Value_3(recieveMessage, startIndex );
                 startIndex += 3;
                 if id ==0 :
                     startIndex = self.BUFFER_SIZE ;
                     continue;

                 distancePerVelocity =  float(recieveMessage[startIndex ]) + float(recieveMessage[startIndex + 1])/ 100;
                 startIndex += 2;

                 distance = float(self.GetBase10Value_3(recieveMessage, startIndex)) / 10;
                 startIndex += 3;

                 self.RAILINE_SIM_ID_DIC[simID] = id;

                 self.RAILLINECOST_DIC[id] = RailLineCost();
                 self.RAILLINECOST_DIC[id].ID = id;

                 self.RAILLINE_DIC[id] = RailLine();
                 self.RAILLINE_DIC[id].ID = id;
                 self.RAILLINE_DIC[id].SimID = simID;
                 self.RAILLINE_DIC[id].DistancePerVelocity= distancePerVelocity;
                 self.RAILLINE_DIC[id].Distance = distance;

                 levelJoiningLineCount = recieveMessage[startIndex] ;
                 self.RAILLINE_DIC[id].LevelJoiningLineCount = levelJoiningLineCount;
                 startIndex += 1;
                 index = 0;
                 while index < levelJoiningLineCount:
                     levelJoiningID = self.GetBase10Value_2(recieveMessage, startIndex );
                     self.RAILLINE_DIC[id].LevelJoiningLineIDList.append(levelJoiningID);
                     startIndex += 2;
                     index += 1;



                 level2JoiningLineCount = recieveMessage[startIndex ] ;
                 startIndex += 1;
                 index = 0 ;
                 self.RAILLINE_DIC[id].Level2JoiningLineCount = level2JoiningLineCount;
                 while index < level2JoiningLineCount:
                     level2JoiningID = self.GetBase10Value_2(recieveMessage, startIndex );
                     self.RAILLINE_DIC[id].Level2JoiningLineIDList.append(level2JoiningID);
                     startIndex += 2;
                     index += 1;

                 level3JoiningLineCount = recieveMessage [startIndex];
                 startIndex = startIndex + 1;
                 index = 0;

                 self.RAILLINE_DIC[id].Level3JoiningLineCount = level3JoiningLineCount;

                 while index < level3JoiningLineCount:
                     level3JoiningID = self.GetBase10Value_2(recieveMessage, startIndex);
                     self.RAILLINE_DIC[id].Level3JoiningLineIDList.append(level3JoiningID);
                     startIndex += 2;
                     index += 1;
                     
                 divergingLineCount  = recieveMessage [startIndex];
                 startIndex+= 1;
                 index= 0;
                 self.RAILLINE_DIC[id].DivergingLineCount = divergingLineCount;
                 if divergingLineCount > 1:
                     self.RAILLINE_DIC[id].LineType = 1;
                 while index < divergingLineCount:
                     divergingID = self.GetBase10Value_2(recieveMessage, startIndex);
                     self.RAILLINE_DIC[id].DivergingLineIDList.append(divergingID);
                     startIndex += 2;
                     index += 1;
                 portCount  = recieveMessage [startIndex];
                 startIndex+= 1;
                 self.RAILLINE_DIC[id].PortCount = portCount;
                 railLineCount = railLineCount + 1;
               else:
                   recieveMessage = self.RecieveMessage(self.BUFFER_SIZE);
                   startIndex = 0;

           ohtCount = 0;

           self.OHT_DIC = {};
           while ohtCount < self.OHT_COUNT:
             if (startIndex + 7 >= self.BUFFER_SIZE):
                 recieveMessage = self.RecieveMessage(self.BUFFER_SIZE);
                 startIndex = 0;
             else :
                 oht_id = self.GetBase10Value_2(recieveMessage, startIndex);

                 if oht_id == 0:
                     startIndex = self.BUFFER_SIZE ;
                     continue;

                 self.OHT_DIC[oht_id] = Oht();
                 self.OHT_DIC[oht_id].ID = oht_id;
                 startIndex += 2;
                 isReticle = recieveMessage[startIndex]
                 self.OHT_DIC[oht_id].RunningAreaType = isReticle;
                 startIndex += 1
                 nameNum = self.GetBase10Value_2(recieveMessage, startIndex)
                 startIndex += 2
                 name = str(nameNum)
                 while len(name) < 5:
                     name = "0"+name
                 
                 if isReticle:
                     name = "R"+name
                 else:
                     name = "V"+name

                 self.OHT_DIC[oht_id].Name = name
                 carrier_types_cnt = recieveMessage[startIndex]
                 startIndex += 1
                 for i in range(carrier_types_cnt):
                     carrier_type = recieveMessage[startIndex]
                     startIndex += 1
                     self.OHT_DIC[oht_id].CarrierTypes.append(carrier_type);
                 
                 ohtCount += 1;

       def SendRailLineCostMessage(self):
           byteArray =  bytearray(self.BUFFER_SIZE);
           current_time = datetime.now();
           dateTimeValue =  current_time.month * 100000000 + current_time.day * 1000000  + current_time.hour * 10000    + current_time.minute * 100  + current_time.second ;

           self.SetByteHexa_4Legnth(dateTimeValue, byteArray , 0);

           index = 4 ;
           count = 0 ;
           for v in self.RAILLINECOST_DIC:
               raillineCost =  self.RAILLINECOST_DIC[v];
               count = count+ 1;
               if index + 6 >= self.BUFFER_SIZE:
                   self.SendMessage(byteArray);
                   byteArray = bytearray(self.BUFFER_SIZE);
                   current_time = datetime.now();
                   dateTimeValue = current_time.month * 100000000 + current_time.day * 1000000 + current_time.hour * 10000 + current_time.minute * 100 + current_time.second;
                   self.SetByteHexa_4Legnth(dateTimeValue, byteArray, 0);
                   index = 4;
               
               self.SetByteHexa_3Legnth(raillineCost.ID, byteArray, index);
               v1 =  math.floor(raillineCost.FRailLineCost);
               if (v == 444):
                   abc= 0 ;
               v2= math.floor(( raillineCost.FRailLineCost * 100) % 100);
               #v2 = (int)((round( raillineCost.FRailLineCost * 100)) % 100);
               self.SetByteHexa_3Legnth(v1, byteArray, index + 3);
               self.SetByteHexa_1Legnth(v2, byteArray, index + 6);
               index = index + 7;

           if index > 4:
               self.SendMessage(byteArray);

           self.WriteLog();

       def GetAssignCommand(self):
           job_list = list()
           recieveMessage = self.RecieveMessage(self.BUFFER_SIZE)
           idx = 0
           command_cnt = self.GetBase10Value_2(recieveMessage, idx)
           idx += 2
           count = 0

           while count < command_cnt:
               if idx + 14 < self.BUFFER_SIZE :
                   command_id = self.GetBase10Value_3(recieveMessage, idx)
                   idx += 3
                   if command_id == 0:
                       recieveMessage = self.RecieveMessage(self.BUFFER_SIZE)
                       idx = 0
                       continue;
             
               from_node = self.GetBase10Value_2(recieveMessage, idx)
               idx += 2
               to_node = self.GetBase10Value_2(recieveMessage, idx)
               idx += 2
               jobState = recieveMessage[idx]
               idx += 1
               ohtId = self.GetBase10Value_2(recieveMessage, idx)
               idx += 2

               carrier_type_cnt = recieveMessage[idx]
               idx += 1
               carrier_type_list = []
               for i in range(carrier_type_cnt):
                    carrier_type = recieveMessage[idx]
                    carrier_type_list.append(carrier_type)
                    idx += 1

               running_Area_type_cnt = recieveMessage[idx]
               idx += 1
               running_Area_type_list = []
               for i in range(running_Area_type_cnt):
                    running_Area_type = recieveMessage[idx]
                    running_Area_type_list.append(running_Area_type)
                    idx += 1

               if self.JOB_DIC.get(command_id):
                  self.JOB_DIC[command_id].FromNode = from_node
                  self.JOB_DIC[command_id].ToNode = to_node
                  self.JOB_DIC[command_id].State = jobState
                  self.JOB_DIC[command_id].OHTId = ohtId
                  self.JOB_DIC[command_id].CarrierTypes = carrier_type_list
                  self.JOB_DIC[command_id].RunningAreaTyes = running_Area_type_list
               else:
                  self.JOB_DIC[command_id] = Job()
                  self.JOB_DIC[command_id].ID = command_id
                  self.JOB_DIC[command_id].FromNode = from_node
                  self.JOB_DIC[command_id].ToNode = to_node
                  self.JOB_DIC[command_id].State = jobState
                  self.JOB_DIC[command_id].OHTId = ohtId
                  self.JOB_DIC[command_id].CarrierTypes = carrier_type_list
                  self.JOB_DIC[command_id].RunningAreaTyes = running_Area_type_list
               job_list.append(self.JOB_DIC[command_id])
               count += 1
           return job_list

       def SendAssignOht(self, oht_dic):
           buffer = bytearray(self.BUFFER_SIZE)
           idx = 0
           self.SetByteHexa_2Legnth(len(oht_dic.keys()),buffer, idx)
           idx += 2

           for job_id, oht_id in oht_dic.items():
               if idx + 5  + len(self.JOB_DIC[job_id].RouteList) * 2 + 2 >= self.BUFFER_SIZE:
                   self.SendMessage(buffer)
                   idx = 0

               self.SetByteHexa_3Legnth(job_id, buffer, idx)
               idx += 3
               self.SetByteHexa_2Legnth(oht_id, buffer, idx)
               idx += 2
               self.SetByteHexa_2Legnth(len(self.JOB_DIC[job_id].RouteList), buffer, idx)
               idx += 2
               for line_id in self.JOB_DIC[job_id].RouteList:
                   self.SetByteHexa_2Legnth(line_id, buffer, idx)
                   idx += 2

           self.SendMessage(buffer)

       def SendConnectMessage(self, deepLearningModelType):
           totalSize = self.MESSAGE_FILE_PATH.__len__() + 8;
           byteArray =  bytearray(self.BUFFER_SIZE);
           current_time = datetime.now();
           dateTimeValue =  current_time.month * 100000000 + current_time.day * 1000000  + current_time.hour * 10000    + current_time.minute * 100  + current_time.second ;

           self.SetByteHexa_4Legnth(dateTimeValue, byteArray, 0);
           self.SetByteHexa_1Legnth(deepLearningModelType, byteArray , 5);
           self.SetByteHexa_2Legnth(self.MESSAGE_FILE_PATH.__len__(), byteArray , 6);

           self.SetByteValueByStringValue(self.MESSAGE_FILE_PATH,byteArray, 8 );

           self.SendMessage(byteArray);

       def SendAllOHTRoute(self):
           idx = 0
           buffer = bytearray(self.BUFFER_SIZE)
           for key in self.OHT_DIC.keys():
              routes = self.OHT_DIC[key].RouteList
              if idx + len(routes)*2 + 3 >= self.BUFFER_SIZE:
                idx = 0
              if idx == 0 and buffer[0] != 0:
                self.SendMessage(buffer)
                buffer = bytearray(self.BUFFER_SIZE)
              self.SetByteHexa_2Legnth(key, buffer, idx)
              idx += 2
              self.SetByteHexa_1Legnth(len(routes), buffer, idx)
              idx += 1
              for route in routes:
                  self.SetByteHexa_2Legnth(route, buffer, idx)
                  idx += 2
           self.SendMessage(buffer)
           return

       def SendSingleRoute(self, key):
           idx = 0
           buffer = bytearray(self.BUFFER_SIZE)
           self.SetByteHexa_2Legnth(key, buffer, idx)
           idx += 2
           routes = self.OHT_DIC[key].RouteList
           self.SetByteHexa_2Legnth(len(routes), buffer, idx)
           idx += 2
           for route in routes:
               if idx + 2 >= self.BUFFER_SIZE:
                   self.SendMessage(buffer)
                   idx = 0
                   buffer = bytearray(self.BUFFER_SIZE)
               self.SetByteHexa_2Legnth(route, buffer, idx)
               idx += 2
           self.SendMessage(buffer)
           return

       def GetNeedReRouteOht(self):
           recieveMessage = self.RecieveMessage(self.BUFFER_SIZE)
           idx = 0
           total_cnt = self.GetBase10Value_2(recieveMessage, idx)
           idx += 2
           oht_list = list()
           i = 0
           from_nodes = list()
           to_nodes = list()
           while i < total_cnt:
               if idx + 7 >= self.BUFFER_SIZE:
                   recieveMessage = self.RecieveMessage(self.BUFFER_SIZE)
                   idx = 0
                   continue;
               oht_id = self.GetBase10Value_2(recieveMessage, idx)
               idx += 2
               if oht_id == 0:
                   recieveMessage = self.RecieveMessage(self.BUFFER_SIZE)
                   idx = 0
                   continue;
               oht_state = recieveMessage[idx]
               idx += 1
               oht_list.append(oht_id)
               from_node = self.GetBase10Value_2(recieveMessage, idx)
               idx += 2
               from_nodes.append(from_node)
               to_node = self.GetBase10Value_2(recieveMessage, idx)
               idx += 2
               to_nodes.append(to_node)
               i += 1
           return oht_list, from_nodes, to_nodes


       def SendReRoute(self,needOHT):
           buffer = bytearray(self.BUFFER_SIZE)
           lastIdx = 0
           self.SetByteHexa_2Legnth(len(needOHT), buffer, lastIdx)
           lastIdx += 2

           for i in needOHT.keys():
               ohtId = i
            
               if lastIdx + 4 + len(needOHT[ohtId]) * 2 >= self.BUFFER_SIZE:
                   self.SendMessage(buffer)
                   lastIdx = 0
                   buffer = bytearray(self.BUFFER_SIZE)
               routes = needOHT[ohtId]
               if (len(self.OHT_DIC[ohtId].RouteList) > len(needOHT[ohtId])) :
                   a = 0;
               elif(len(self.OHT_DIC[ohtId].RouteList) < len(needOHT[ohtId])) : 
                   b=0;
               self.SetByteHexa_2Legnth(ohtId, buffer, lastIdx)
               lastIdx += 2
               self.SetByteHexa_2Legnth(len(routes), buffer,lastIdx);
               lastIdx += 2
               for i in range(len(routes)):
                   self.SetByteHexa_2Legnth(routes[i], buffer , lastIdx);
                   lastIdx += 2

           self.SendMessage(buffer)

       def GetHashCode(self, bufferSize, byteArr):
           hashValue = self.GetHashValue(byteArr);
           hashCode = bytearray(bufferSize);
           self.SetByteHexa_2Legnth(int (hashValue), hashCode,0);
           return hashCode;

       def GetHashValue(self, byteArr):
          index = 0 ;
          l = byteArr.__len__();
          returnValue = int(0);
          while (index < l):
              returnValue +=byteArr[index];
              index = index+ 1;
          return  returnValue;

       def SetByteBase2(selfx, value, byteArr, startIndex):
            quotient = value;
            l = byteArr.__len__();
            index = 0;
            while index < l:
                theRest = int(quotient % 256);
                quotient = math.floor(quotient/ 256);

                byteArr[startIndex + index] = theRest;
                if quotient ==0:
                   break
                index= index+ 1;

            return byteArr;

       def RecieveMessage(self, bufferSize):
          rebufferSize = bufferSize;
          returnData  =bytes();

          count = 0;
          waited = 0;
          while(rebufferSize !=  0 ):
            if (count >= 1):
                b = 0;
            try:
                data =  self.client_socket.recv(rebufferSize)
            except socket.timeout:
                # recv 무응답. 어느 recv가 몇 바이트 받고 멈췄는지 기록.
                waited += self.RECV_TIMEOUT;
                got = bufferSize - rebufferSize;
                hangmsg = ("[HANG] RecieveMessage 대기 " + str(waited) + "s — 요청 "
                           + str(bufferSize) + "B 중 " + str(got) + "B 수신 후 멈춤 (마지막 v="
                           + str(self.LAST_V) + ", 세션누적수신=" + str(self.TOTAL_BYTES_READ) + "B).");
                print(hangmsg);
                try:
                    self.WriteAdminLog(hangmsg);
                except Exception:
                    pass
                if waited >= 180:
                    # 180s 무응답 = 진짜 hang(대개 desync로 시뮬이 더 못 보내는 상태). 영원히 붙잡지 말고
                    # 예외 → main의 except가 소켓 닫고 재접속(동료 코드처럼 회복).
                    raise RuntimeError("RecieveMessage 180s 무응답 — desync 의심, 재접속 (마지막 v=" + str(self.LAST_V) + ")")
                continue;
            if not data:
                # recv()가 b''를 반환하면 peer가 연결을 정상 종료한 것이다.
                # 부분/빈 버퍼를 이후 파서에 넘기면 IndexError로 원인이 가려지므로,
                # 즉시 예외를 올려 main의 재접속 경로를 타게 한다.
                got = bufferSize - rebufferSize
                raise ConnectionError(
                    "Socket peer disconnected while receiving "
                    + str(bufferSize) + "B message (received " + str(got) + "B)"
                )

            rebufferSize = rebufferSize -len(data)
            returnData =returnData + data ;
            self.TOTAL_BYTES_READ += len(data);
            count + 1;
          if (count >= 2):
              self.secData["Count 3 통신 Error"] = count;
          return returnData;

       def PeekPending(self, n=256):
          # unknown v 진단: 스트림에 즉시 대기중인 바이트를 소비하지 않고(MSG_PEEK) 들여다본다.
          # 대기 바이트가 있으면 = 직전 recv가 스트림을 덜 읽어 desync 발생(잔여물). 없으면 = 단독 패킷.
          try:
              self.client_socket.setblocking(False);
              try:
                  return self.client_socket.recv(n, socket.MSG_PEEK);
              except (BlockingIOError, OSError):
                  return b'';
              finally:
                  self.client_socket.settimeout(self.RECV_TIMEOUT);
          except Exception:
              return b'';

       def SendMessage(self, message):
           self.client_socket.sendall(message);

       def IsSame(self, byteArr1, byteArr2):
           if byteArr1.__len__() != byteArr2.__len__() :
               return False;

           index= 0;
           l= byteArr1.__len__();
           while index < l :
               if byteArr1[index] != byteArr2[index]:
                   return False;
               index = index + 1;

           return True;
            
       def SetByteHexa_4Legnth(self, value, byteArr, startIndex):
           if (value > 4294967296):
                value = 4294967296;
           
           byteArr[startIndex + 3] = (value & 0x000000ff);
           byteArr[startIndex + 2] = ((value & 0x0000ff00) >> 8);
           byteArr[startIndex + 1] = ((value & 0x00ff0000) >> 16);
           byteArr[startIndex] = ((value & 0xff000000) >> 24);

       def SetByteHexa_3Legnth(self, value, byteArr, startIndex):
           if (value > 16777216):
                value = 16777216;
           byteArr[startIndex + 2] = (value & 0x000000ff);
           byteArr[startIndex + 1] = ((value & 0x0000ff00) >> 8);
           byteArr[startIndex] = ((value & 0x00ff0000) >> 16);

       def SetByteHexa_2Legnth(self, value, byteArr, startIndex):
           if (value > 65535):
                value = 65535;
           byteArr[startIndex + 1] = (value & 0x000000ff);
           byteArr[startIndex ] = ((value & 0x0000ff00) >> 8);

       def SetByteHexa_1Legnth(self, value, byteArr, startIndex):
           if (value > 255):
                value = 255;
           byteArr[startIndex ] = (value & 0x000000ff);

       def SetByteValueByStringValue(self, value ,byteArr, startIndex ):
           index = startIndex;
           for char in value:
               byteArr[index]  = char.encode()[0];
               index = index + 1;

       def GetBase10Value_2(self, byteArr, startIndex):
           returnValue = 0;
           returnValue += byteArr[startIndex + 1] ;
           returnValue += byteArr[startIndex ] <<8;

           return  returnValue;
      
       def GetBase10Value_3(self, byteArr, startIndex):
           returnValue = 0;
           returnValue += byteArr[startIndex + 2] ;
           returnValue += byteArr[startIndex + 1] <<8;
           returnValue += byteArr[startIndex ] <<16;
           return  returnValue;

       def GetBase10Value_4(self, byteArr, startIndex):
           returnValue = 0;
           returnValue += byteArr[startIndex + 3] ;
           returnValue += byteArr[startIndex + 2] <<8 ;
           returnValue += byteArr[startIndex + 1] <<16;
           returnValue += byteArr[startIndex ] <<24;
           return  returnValue;
