"""Internal baseline/inference server entry point for run_headless.py."""
from oht_routing.utils.wandb_logging import EXP_META

EXP_META.update(
    note="headless", reward_version="Q",
    description=("GUI-free Pinokio batch evaluation through the original TCP "
                 "protocol. Terminal JSONL uses the last active packet; "
                 "simulator result databases are retained per episode."),
)

if __name__ == "__main__":
    from main import main
    main()
