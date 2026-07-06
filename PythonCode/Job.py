class Job:
    # Job ID
    ID  = 0 ;

    # Job State
    #  1: QUEUED,    Command 생성된 이후에 oht에 할당되기 전 상태
    #  2: RESERVATED,다른 Command를 Transferring 중인 oht가 일을 마치고 해당 Command를 수행하도록 예비 할당된 상태
    #  3: WAITING,   Foup을 가지러 가는 상태
    #  5: TRANSFERRING,Foup을 내려 놓으러 가는 상태
    #  7: COMPLETED 반송 완료 상태
    State = 0 ;

    # Schedule에 사용 되는 우선순위 Parameter
    Priority = 0;

    # 경로 RailLine id
    RouteList = [];

    # Foup을 가지가는 경로 및 통과 시간
    Waiting_PassLines = [];
    


     # Foup을 내려놓으러 가는 경로 및 통과 시간
    Transfer_PassLines = [];

    # Equipment 여부  1: Equipment, 0: Non Equipment
    IsEqp = 0;

    # Job을 수행하는 OHT들이 바뀐 횟수
    ReAssignCount = 0 ;

    # Job의 FromNode의 ID
    FromNode = 0

    # Job의 ToNode의 ID
    ToNode = 0

    # Job의 OHT ID
    OHTId = 0

    # Job의 CarrierType
    CarrierTypes = []

    # Job의 RunningAreaType
    RunningAreaTyes = []


    def __init__(self):
        self.ID = -1;
        self.State = -1;
        self.Priority = -1;
        self.RouteList = [];
        self.IsEqp = 0 ;
        self.Waiting_PassLines = [];
        self.Transfer_PassLines = [];
        self.ReAssignCount =1 ;
        self.FromNode = -1
        self.ToNode = -1
        self.OHTId = 0
        self.CarrierTypes = []
        self.RunningAreaTyes = []