"""Entry point for contextual baseline, inference, and training."""

import sys

from oht_routing.runtime.client import ClientAlgorithm
from oht_routing.runtime.cli import parse_args
# Public imports retained for protocol tests and existing launcher integrations.
from oht_routing.runtime.protocol import (
    handle_command,
    read_port,
    send_active_data,
)
from oht_routing.runtime.config import (
    runtime_config_from_args,
    seed_everything,
)
from oht_routing.runtime.summary import print_runtime_summary
from oht_routing.runtime.server import serve_contextual


def _configure_console_encoding():
    try:
        sys.stdout.reconfigure(
            encoding="utf-8", errors="replace", line_buffering=True
        )
        sys.stderr.reconfigure(
            encoding="utf-8", errors="replace", line_buffering=True
        )
    except (AttributeError, ValueError):
        pass


def main():
    _configure_console_encoding()
    args = parse_args()
    if args.sim_end_time <= 0:
        raise ValueError("--sim-end-time must be positive")
    if args.console_log_interval <= 0:
        raise ValueError("--console-log-interval must be positive")

    config = runtime_config_from_args(args)
    seed_everything(config.seed)
    client = ClientAlgorithm(config)
    print_runtime_summary(args, client)
    serve_contextual(args, client)


if __name__ == "__main__":
    main()
