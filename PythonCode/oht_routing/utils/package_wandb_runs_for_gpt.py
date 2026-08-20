"""Package selected W&B histories as labeled CSV files in one ZIP archive."""

import csv
import io
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WANDB_RESULTS = ROOT / "results" / "wandb"
OUTPUT = WANDB_RESULTS / "contextual_td7_4runs_labeled_for_gpt.zip"
RUN_IDS = ("f8u5by9y", "zuw9nno8", "0lwv8bat", "26hxsuet")

SAMPLING_LABELS = {
    "f8u5by9y": "REPLAY_SAMPLING_RAIL",
    "zuw9nno8": "REPLAY_SAMPLING_RANDOM_RAIL",
    "0lwv8bat": "LEGACY_UNSPECIFIED",
    "26hxsuet": "REPLAY_SAMPLING_RAIL",
}

LABEL_COLUMNS = (
    "label/run_id",
    "label/replay_sampling",
    "label/reward_version",
    "label/tat_confidence_ramp",
    "label/tat_confidence_n0",
    "label/batch_size",
    "label/dispatch_mode_effective",
    "label/dispatch_mode_configured",
    "label/dispatch_selection_version",
    "label/device",
    "label/resumed_from_checkpoint",
    "label/run_state_at_download",
    "label/snapshot_env_step",
)


def labels_for(run_id, info):
    config = info["config"]
    meta = config.get("EXP_META", {})
    resume_path = config.get("resume_checkpoint_path")
    return {
        "label/run_id": run_id,
        "label/replay_sampling": SAMPLING_LABELS[run_id],
        "label/reward_version": meta.get("reward_version", ""),
        "label/tat_confidence_ramp": config.get("tat_confidence_ramp", False),
        "label/tat_confidence_n0": config.get("tat_confidence_n0", ""),
        "label/batch_size": config.get("batch_size", ""),
        # Cost dispatch did not exist for the legacy runs; the prior behavior
        # was retained as the later first-match default.
        "label/dispatch_mode_effective": "first-match",
        "label/dispatch_mode_configured": config.get("dispatch_mode", ""),
        "label/dispatch_selection_version": meta.get(
            "dispatch_selection_version", "legacy_pre_versioned"
        ),
        "label/device": config.get("device", ""),
        "label/resumed_from_checkpoint": bool(resume_path),
        "label/run_state_at_download": info.get("state", ""),
        "label/snapshot_env_step": info.get("summary", {}).get(
            "env/step", info.get("summary", {}).get("_step", "")
        ),
    }


def write_labeled_csv(archive, run_id, labels):
    source = WANDB_RESULTS / run_id / "history.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as input_stream:
        reader = csv.DictReader(input_stream)
        member = f"{run_id}_labeled.csv"
        with archive.open(member, "w") as binary_stream:
            with io.TextIOWrapper(
                binary_stream, encoding="utf-8-sig", newline=""
            ) as output_stream:
                writer = csv.DictWriter(
                    output_stream,
                    fieldnames=list(LABEL_COLUMNS) + list(reader.fieldnames or ()),
                )
                writer.writeheader()
                for row in reader:
                    writer.writerow({**labels, **row})


def main():
    manifests = []
    with zipfile.ZipFile(
        OUTPUT,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for run_id in RUN_IDS:
            info_path = WANDB_RESULTS / run_id / "run_info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            labels = labels_for(run_id, info)
            write_labeled_csv(archive, run_id, labels)
            manifests.append(labels)
            print(f"Packaged {run_id}", flush=True)

        manifest_stream = io.StringIO(newline="")
        writer = csv.DictWriter(manifest_stream, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(manifests)
        archive.writestr("manifest.csv", "\ufeff" + manifest_stream.getvalue())

        archive.writestr(
            "README.txt",
            "Each CSV contains the original W&B history plus label/* columns.\n"
            "zuw9nno8 is the only REPLAY_SAMPLING_RANDOM_RAIL run.\n"
            "LEGACY_UNSPECIFIED means replay_sampling_mode was not recorded in "
            "that older run's W&B config; it is intentionally not guessed.\n"
            "All runs use effective first-match dispatch, but their dispatch "
            "implementation versions can differ.\n",
        )

    print(OUTPUT)


if __name__ == "__main__":
    main()
