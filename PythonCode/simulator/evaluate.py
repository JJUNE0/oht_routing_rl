"""Evaluate simulator SQLite results against the configured baseline."""

import numpy as np
import pandas as pd 
import sqlite3
import os 
from typing import Tuple
import argparse
from contextlib import closing
from pathlib import Path

#####################################################################################################################################
#####################################################사용자 변경 부분#################################################################
#####################################################################################################################################
# DB 파일 위치 (기본값: 이 스크립트 기준 ../results). 필요 시 절대경로로 교체해 사용.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
root_file_dir = os.path.join(PROJECT_ROOT, "results")
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


def _read_result_frame(database):
    path = Path(database).resolve()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        result = pd.read_sql_query("SELECT * FROM COMMAND_LOG", conn)
    for column in ("ACTIVATED_TIME", "ASSIGNED_TIME", "COMPLETED_TIME"):
        result[column] = pd.to_datetime(result[column], errors="coerce")
    return result


def _window_metrics(result, start_datetime, end_datetime):
    selected = result[(result.ACTIVATED_TIME > start_datetime) &
                      (result.ACTIVATED_TIME < end_datetime)]
    oht_count = result["OHT_NAME"].nunique()
    return {
        "count": len(selected),
        "tat": tat(result, start_datetime, end_datetime),
        "operation_rate": operation_rate(result, start_datetime, end_datetime, oht_count)
        if oht_count else float("nan"),
        # Timestamp coverage is necessary but is not a full-rollout guarantee.
        "reaches_window_end": bool(result.COMPLETED_TIME.max() >= end_datetime),
    }


def evaluate(args):
    file_path = Path(args.save_dir) / args.db_filename
    start_datetime = pd.Timestamp(getattr(args, "start_time", None) or start_time)
    end_datetime = pd.Timestamp(getattr(args, "end_time", None) or end_time)
    if pd.isna(start_datetime) or pd.isna(end_datetime) or end_datetime <= start_datetime:
        raise ValueError("Evaluation end time must be after start time")
    current = _window_metrics(_read_result_frame(file_path), start_datetime, end_datetime)
    baseline_db = getattr(args, "baseline_db", None)
    baseline = None
    source = None
    if baseline_db:
        baseline = _window_metrics(_read_result_frame(baseline_db), start_datetime, end_datetime)
        if not baseline["reaches_window_end"] or not np.isfinite(baseline["tat"]) or baseline["tat"] <= 0:
            raise ValueError("Baseline DB must contain valid completed samples and reach the evaluation end")
        source = str(Path(baseline_db).resolve())
    elif start_datetime == pd.Timestamp(start_time) and end_datetime == pd.Timestamp(end_time):
        baseline = {"count": trcnt_total_base, "tat": tat_total_base,
                    "operation_rate": oht_operation_rate_base}
        source = "legacy evaluate.py constants for the 09:00-19:00 window"

    comparable = baseline is not None and current["reaches_window_end"]
    count_improvement = (current["count"] - baseline["count"]) / baseline["count"] if comparable else None
    tat_improvement = (baseline["tat"] - current["tat"]) / baseline["tat"] if comparable else None
    operation_difference = baseline["operation_rate"] - current["operation_rate"] if comparable else None
    rendered = lambda value: "unavailable" if value is None else str(round(value, 5))
    print(f"SimulationResult DB filepath : {file_path}")
    print(f"evaluation window : {start_datetime} < ACTIVATED_TIME < {end_datetime}")
    print(f"window hours : {(end_datetime - start_datetime).total_seconds() / 3600:g}")
    print(f"baseline source : {source or 'not supplied for this window; no improvement claim'}")
    if not current["reaches_window_end"]:
        print("PARTIAL: completed timestamps do not reach the evaluation end; no improvement claim")
    print(f"trcnt_total_new : {current['count']}, improvement_rate : {rendered(count_improvement)}")
    print(f"tat_total_new : {round(current['tat'],5)}, improvement_rate : {rendered(tat_improvement)}")
    print(f"oht_operation_rate_new : {round(current['operation_rate'],5)}, difference : {rendered(operation_difference)}")
    return {
        "tat": current["tat"], "operation_rate": current["operation_rate"],
        "command_count": current["count"], "tat_s": current["tat"] * 60,
        "window_start": str(start_datetime), "window_end": str(end_datetime),
        "reaches_window_end": current["reaches_window_end"],
        "baseline_source": source, "baseline_tat_s": baseline["tat"] * 60 if baseline else None,
        "tat_improvement_rate": tat_improvement,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_results = os.path.join(
        PROJECT_ROOT, "results", "contextual_td7", "0820"
    )
    parser.add_argument('--save_dir', type=str, default=default_results, help="결과 DB가 들어있는 폴더")
    parser.add_argument('--db_filename', type=str, default='ep_1.db', help="평가할 결과 DB 파일명")
    parser.add_argument('--start-time', help="ACTIVATED_TIME 하한, 예: 2026-02-16 07:00:00")
    parser.add_argument('--end-time', help="ACTIVATED_TIME 상한, 예: 2026-02-16 19:00:00")
    parser.add_argument('--baseline-db', type=Path, help="같은 시간창으로 다시 계산할 완료 baseline DB")
    args = parser.parse_args()
    

    evaluate(args)
