


class RailLine:
    ID = 0 ;
    SimID = 0;
    # 1sec 당 이동하는 mm
    DistancePerVelocity= 0 ;

    # RailLine 길이  (mm)
    Distance = 0 ;
    # RailLine으로 들어오는 Line의 수
    LevelJoiningLineCount = 0;
    # RailLine으로 들어오는 Line의 ID List
    LevelJoiningLineIDList = [];
    # RailLine으로 들어오는 Line의 Line의 수
    Level2JoiningLineCount = 0 ;
    # RailLine으로 들어오는 Line의 Line의 ID List
    Level2JoiningLineIDList = [];
    # RailLine으로 들어오는 Line의 Line의 Line의 수
    Level3JoiningLineCount = 0;
    # RailLine으로 들어오는 Line의 Line의 Line의 ID List
    Level3JoiningLineIDList = [];
    # RailLine에서 나가는 Line의 수
    DivergingLineCount = 0;
    # RailLine에서 나가는 Line의 ID List
    DivergingLineIDList = [];
    # 분기점 유무, 1일 경우 분기점
    LineType = 0;
    # RailLine으로 있는 OHT들  Index가 낮을 수록 앞에 있는 Oht
    OhtList = [];

    PredictedOHTIDList= [];

    IdleOHTCount = 0;
    ReservationPortCount = 0;
    PortCount =0;

    SimAvgSpeed = 0

    def __init__(self):
        self.ID = 0;
        self.SimID = 0;
        self.DistancePerVelocity = 0;
        self.LineWeight = 0 ;
        self.Distance = 0;
        self.LevelJoiningLineCount = 0;
        self.LevelJoiningLineIDList = [];
        self.Level2JoiningLineCount = 0;
        self.Level3JoiningLineCount = 0;
        self.PredictedOHTCount = 0;
        self.Level2JoiningLineIDList = [];
        self.Level3JoiningLineIDList = [];
        self.DivergingLineCount = 0 ;
        self.DivergingLineIDList = [] ;
        self.OhtList = [];
        self.IdleOHTCount = 0 ;
        self.ReservationPortCount = 0 ;
        self.PortCount = 0;
        self.SimAvgSpeed = 0



