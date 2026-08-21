"""Download a Weights & Biases run's metrics for local analysis.

Example:
    python download_wandb_run.py
    python download_wandb_run.py --run-path bjy6614-postech/oht-routing-rl-td7-region/5ytouc3j
    python download_wandb_run.py --keys global/tat reward/step_reward loss/q1

Authentication:
    Run ``wandb login`` once, or set the WANDB_API_KEY environment variable.
"""

import argparse
import csv
import json
from pathlib import Path

import wandb

DEFAULT_RUN_PATH = "bjy6614-postech/oht-routing-rl-td7-region/w5l4ytzi"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Download W&B run history as CSV and JSON.")
    parser.add_argument("--run-path", default=DEFAULT_RUN_PATH, help="entity/project/run_id")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory (default: results/wandb/<run_id>).",
    )
    parser.add_argument(
        "--keys",
        nargs="*",
        default=None,
        help="Optional history keys to download; omit to retrieve all scalar history keys.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000,
        help="Print download progress every N rows (default: 1000).",
    )
    return parser.parse_args()


def json_default(value):
    """Keep uncommon W&B values serializable without losing the row."""
    return str(value)


def main():
    args = parse_args()
    run_id = args.run_path.rstrip("/").split("/")[-1]
    output_dir = args.output_dir or PROJECT_ROOT / "results" / "wandb" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    run = api.run(args.run_path)
    print(f"Downloading {run.path} ({run.name}) -> {output_dir}")

    run_info = {
        "path": run.path,
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "url": run.url,
        "created_at": str(run.created_at),
        "config": dict(run.config),
        "summary": dict(run.summary),
    }
    (output_dir / "run_info.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )

    jsonl_path = output_dir / "history.jsonl"
    row_count = 0
    all_keys = set()
    # Save each page immediately: a long history download can take minutes and should
    # remain useful even if the user stops it before the CSV conversion completes.
    with jsonl_path.open("w", encoding="utf-8") as file:
        for row in run.scan_history(keys=args.keys, page_size=1_000):
            cleaned_row = {
                key: json_default(value) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
            file.write(json.dumps(cleaned_row, ensure_ascii=False, default=json_default) + "\n")
            all_keys.update(cleaned_row)
            row_count += 1
            if row_count % max(1, args.progress_every) == 0:
                print(f"Downloaded {row_count:,} history rows...")

    if row_count == 0:
        print("No history rows were returned. Metadata was saved to run_info.json.")
        return

    all_keys = sorted(all_keys)
    print("Converting JSONL history to CSV...")
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        with jsonl_path.open("r", encoding="utf-8") as jsonl_file:
            for line in jsonl_file:
                writer.writerow(json.loads(line))

    print(f"Saved {row_count:,} history rows and {len(all_keys):,} columns.")
    print(f"CSV: {output_dir / 'history.csv'}")


if __name__ == "__main__":
    main()
