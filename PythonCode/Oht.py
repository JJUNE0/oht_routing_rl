
class Oht:
    # OHT의 ID
    ID = 0

    # OHT의 State
    # 0: IDLE, 1: STAGE, 2: MOVE_TO_LOAD, 3: LOADING, 4: MOVE_TO_UNLOAD, 5:
    # UNLOADING, 6: NULL 6은 사용 x
    State = 0

    # RailLine 시작점에서 OHT의 현재 위치까지의 거리
    CurrentDistance = -1

    # OHT의 Job ID 없을 경우 0
    JobID = 0

    # OHT의 목적 RailLine ID
    DestinationLine = -1

    # 현재 까지 OHT가 Idle 상태인 시간
    IdleTime = -1

    # Loading, Unloading을 제외한 정체 시간
    StopTime = -1

    # OHT가 통과할 RailLine ID
    RouteList = []
      
    # OHT가 통과한 Line과 시간
    PassTimes = []

    FrontOhts = {}

    # OHT가 통과 중인 Line과 시간
    NotPassTimes = []

    # OHT가 통신 중 움직인 거리
    PassDistance = 0

    # 통신 사이, OHT의 Line별 속도
    VelByLine = {}

    Short_PassDistance = []

    # Simulation 상 OHT의 이름
    Name = ""

    NotPassTimeTime = []

    # OHT Loading / UnLoading 완료까지 남은 시간
    RemainTime = 0

    CarrierTypes = []

    RunningAreaType = 0

    DispatchedCommand = 0

    OperationRate = 0

    CmdCompleteTat = {}

    OhtWorkTimeByCmd = 0.0

    OhtIndividualTat = 0.0 # 4번

    OperationTAT = 0

    RemainingDistanace = 0

    def __init__(self):
        self.ID = -1
        self.State = -1
        self.CurrentDistance = -1
        self.JobID = 0
        self.IdleTime = -1
        self.StopTime = -1
        self.RouteList = []
        self.PassTimes = []
        self.NotPassTimes = []
        self.VelByLine = {}
        self.FrontOhts = {}
        self.Name = ""
        self.Short_PassDistance = []
        self.NotPassTimeTime = []
        self.RemainTime = 0
        self.CarrierTypes = []
        self.RunningAreaType = -1
        self.DispatchedCommand = 0
        self.OperationRate = 0
        self.CmdCompleteTat = {}
        self.OhtWorkTimeByCmd = 0.0
        self.OhtIndividualTat = 0.0
        self.OperationTAT = 0 ;
        self.RemainingDistanace = 0;