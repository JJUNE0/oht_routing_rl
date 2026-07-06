

class ohtLinePassTime:
    # Line ID
    ID = 0 ;
    # 통과 시간
    PassTime = 0;

    # 0: IDLE,  1: STAGE, 2: MOVE_TO_LOAD, 3: LOADING, 4: MOVE_TO_UNLOAD, 5: UNLOADING, 6: NULL    6은 사용 x
    State = 0 ;

    # Line의 시작점으로 부터 이동한 거리
    Distance = 0 ;
