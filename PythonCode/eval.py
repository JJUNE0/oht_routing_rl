import numpy as np
import pandas as pd 
import sqlite3
import os 
from typing import Tuple
import torch
import argparse

#####################################################################################################################################
#####################################################사용자 변경 부분#################################################################
#####################################################################################################################################
# DB 파일 위치 (기본값: 이 스크립트 기준 ../results). 필요 시 절대경로로 교체해 사용.
root_file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
db_file_dir = r'' # 예: r'SimulationResult_3_13_9_50_55.db'
file_path = os.path.join(root_file_dir,db_file_dir)

# 결과파일 저장경로
csv_name = db_file_dir.split('.')[0]
csv_file_path = f'./{csv_name}.csv'

start_time = '2026-02-16 09:00:00'
end_time = '2026-02-16 19:00:00'

trcnt_total_base        = 175299
tat_total_base          = 2.90706
oht_operation_rate_base = 0.79292

#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################

#score 연산 
def tat(df: pd.DataFrame, start_datetime:pd.Timestamp, end_datetime:pd.Timestamp) -> Tuple[np.float64, np.float64]:
    
    mask_total = ((df.ACTIVATED_TIME > start_datetime) & (df.ACTIVATED_TIME < end_datetime))
    tat = (df[mask_total]['COMPLETED_TIME'] - df[mask_total]['ACTIVATED_TIME']).dt.total_seconds() / 60

    return np.mean(tat)


def operation_rate(df: pd.DataFrame, start_datetime:pd.Timestamp , end_datetime:pd.Timestamp, oht_count: int) -> float:
    
    df = df[(df.COMPLETED_TIME > start_datetime) & (df.ACTIVATED_TIME < end_datetime) ]
    
    operation_time = 0.0

    for idx, row in df.iterrows():
        assigned = row['ASSIGNED_TIME']
        completed = row['COMPLETED_TIME']

        op_start_time = max(assigned, start_datetime )
        op_end_time = min(completed, end_datetime )

        operation_time += (op_end_time - op_start_time).total_seconds()
        
        
    total_interval_seconds = (end_datetime - start_datetime).total_seconds()
    operation_rate = operation_time / (total_interval_seconds * oht_count)
    
    return operation_rate

def main():
    # DB 연결
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor() # 연결 생성
    cursor.execute("SELECT * FROM COMMAND_LOG")
    res = [list(row) for row in cursor.fetchall()]
    cursor.execute("PRAGMA table_info(COMMAND_LOG)")
    col_name_list = [column[1] for column in cursor.fetchall()]

    result = pd.DataFrame(res,columns=col_name_list)

    #start/end time 
    start_datetime = pd.to_datetime(start_time)
    end_datetime = pd.to_datetime(end_time)

    time_columns = [col for col in result.columns if "TIME" in col.upper()]
    for col in time_columns:
        result[col] = pd.to_datetime(result[col], errors = 'coerce')
    
    #score 연산
    trcnt_total_new = result[(result.ACTIVATED_TIME > start_datetime) & (result.ACTIVATED_TIME < end_datetime) ].shape[0]
    oht_count = len(np.unique(result['OHT_NAME']))
    tat_total_new = tat(result, start_datetime, end_datetime)
    oht_operation_rate_new = operation_rate(result, start_datetime, end_datetime, oht_count)

    trcnt_total_improvement_rate = (trcnt_total_new - trcnt_total_base)/ trcnt_total_base
    tat_total_improvement_rate =  (tat_total_base - tat_total_new) / tat_total_base
    oht_operation_rate_difference = (oht_operation_rate_base - oht_operation_rate_new)
    
      
    print(f'SimulationResult DB filepath : {db_file_dir}')
    print(f'trcnt_total_new : {trcnt_total_new}, improvement_rate : {round(trcnt_total_improvement_rate,3)}')

    print(f'tat_total_new : {round(tat_total_new,3)}, improvement_rate : {round(tat_total_improvement_rate,3)}')
    print(f'oht_operation_rate_new : {round(oht_operation_rate_new,3)}, difference : {round(oht_operation_rate_difference,3)}')


def evaluate(args):
    # DB 연결
    file_path = os.path.join(args.save_dir, args.db_filename)

    conn = sqlite3.connect(file_path)
    cursor = conn.cursor() # 연결 생성
    cursor.execute("SELECT * FROM COMMAND_LOG")

    res = [list(row) for row in cursor.fetchall()]
    cursor.execute("PRAGMA table_info(COMMAND_LOG)")
    col_name_list = [column[1] for column in cursor.fetchall()]

    result = pd.DataFrame(res,columns=col_name_list)

    #start/end time 
    start_datetime = pd.to_datetime(start_time)
    end_datetime = pd.to_datetime(end_time)
    
    time_columns = [col for col in result.columns if "TIME" in col.upper()]
    for col in time_columns:
        result[col] = pd.to_datetime(result[col], errors = 'coerce')

    trcnt_total_new = result[(result.ACTIVATED_TIME > start_datetime) & (result.ACTIVATED_TIME < end_datetime) ].shape[0]
    oht_count = len(np.unique(result['OHT_NAME']))
    
    tat_total_new = tat(result, start_datetime, end_datetime)
    oht_operation_rate_new = operation_rate(result, start_datetime, end_datetime, oht_count)

    trcnt_total_improvement_rate = (trcnt_total_new - trcnt_total_base)/ trcnt_total_base
    tat_total_improvement_rate =  (tat_total_base - tat_total_new) / tat_total_base
    oht_operation_rate_difference = (oht_operation_rate_base - oht_operation_rate_new)

    print(f'SimulationResult DB filepath : {file_path}')
    print(f'trcnt_total_new : {trcnt_total_new}, improvement_rate : {round(trcnt_total_improvement_rate,5)}')

    print(f'tat_total_new : {round(tat_total_new,5)}, improvement_rate : {round(tat_total_improvement_rate,5)}')
    print(f'oht_operation_rate_new : {round(oht_operation_rate_new,5)}, difference : {round(oht_operation_rate_difference,5)}')

    

    return {"tat" : tat_total_new, "operation_rate" : oht_operation_rate_new}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_results = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results/contextual_td7/0806")
    parser.add_argument('--save_dir', type=str, default=default_results, help="결과 DB가 들어있는 폴더")
    parser.add_argument('--db_filename', type=str, default='ep_3.db', help="평가할 결과 DB 파일명")
    args = parser.parse_args()
    

    evaluate(args)