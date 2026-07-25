"""Export the contextual training metric schema from one W&B run to CSV."""

import argparse
import csv

from PythonCode.contextual_wandb import EXP_META, WANDB_METRIC_KEYS


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
    with open(args.output, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in EXPORT_COLUMNS})


if __name__ == "__main__":
    main()
