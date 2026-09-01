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
    config = runtime_config_from_args(args)
    seed_everything(config.seed)
    if config.num_sim > 1:
        from oht_routing.runtime.distributed.server import serve_distributed

        serve_distributed(config)
        return
    client = ClientAlgorithm(config)
    print_runtime_summary(client)
    port = config.sim_ports[0] if config.sim_ports is not None else None
    serve_contextual(client, port=port)


if __name__ == "__main__":
    main()
