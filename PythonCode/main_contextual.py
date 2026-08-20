"""Thin entry point for contextual baseline, inference, and training."""

import sys

from ClientAlgorithm_contextual import ClientAlgorithm
from contextual_cli import parse_args
# Public imports retained for protocol tests and existing launcher integrations.
from contextual_protocol_runtime import (
    SmokeReporter,
    handle_command,
    read_port,
    send_active_data,
)
from contextual_runtime_bootstrap import (
    runtime_config_from_args,
    seed_everything,
)
from contextual_runtime_summary import print_runtime_summary
from contextual_server import serve_contextual


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
    reporter = (
        SmokeReporter(
            args.smoke_report, args.mode, args.smoke_report_interval
        )
        if args.smoke_report is not None
        else None
    )
    print_runtime_summary(args, client)
    serve_contextual(args, client, reporter)


if __name__ == "__main__":
    main()
