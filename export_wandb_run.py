"""Export the contextual training metric schema from one W&B run to CSV."""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_CODE_DIR = PROJECT_ROOT / "PythonCode"
if str(PYTHON_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODE_DIR))

from contextual_wandb import EXP_META, WANDB_METRIC_KEYS


TEMPORAL_ACTION_COLUMNS = (
    "action/policy_temporal_delta_mean",
    "action/policy_temporal_delta_std",
    "action/exploratory_temporal_delta_mean",
    "action/exploratory_temporal_delta_std",
    "action/exploration_noise_std",
)
if not set(TEMPORAL_ACTION_COLUMNS).issubset(WANDB_METRIC_KEYS):
    raise RuntimeError("temporal action diagnostics are missing from W&B schema")

EXPORT_COLUMNS = ("_step",) + WANDB_METRIC_KEYS
EXPORT_SCHEMA_VERSION = EXP_META["diagnostic_schema_version"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", help="entity/project/run_id")
    parser.add_argument("--output", default="contextual_wandb_run.csv")
    args = parser.parse_args()
    import wandb

    run = wandb.Api().run(args.run_path)
    rows = run.scan_history(keys=list(EXPORT_COLUMNS))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in EXPORT_COLUMNS})


if __name__ == "__main__":
    main()
