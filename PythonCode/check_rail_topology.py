"""One-shot live topology audit for contextual per-rail TD7.

Start this script first, then start the simulator. It receives only the
simulator initialization payload, writes the fixed 10-in/10-out mapping audit
and cache, and exits without creating a learning algorithm.
"""

import argparse
import socket
import sys
from datetime import datetime
from pathlib import Path

import PClient
from oht_routing.mdp.topology import TopologyAuditError, build_contextual_topology


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9100
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent
EXPECTED_SIMULATION_STATES = {0, 1, 2, 3, 4, 5, 6}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit live rail topology for contextual TD7 and exit without training."
        )
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for topology_neighbor_audit.json and topology cache.",
    )
    return parser.parse_args()


def wait_for_initialized_topology(pclient):
    """Wait through the simulator's zero-rail handshake until live rails arrive."""
    while (
        int(pclient.RAILINE_COUNT) == 0
        and len(pclient.RAILLINE_DIC) == 0
    ):
        timestamp = datetime.now().strftime("%Y.%m.%d - %H:%M:%S")
        print(timestamp)
        print("[topology-audit] waiting for initialized rail topology")
        try:
            pclient.WriteAdminLog(
                "Topology audit: waiting for initialized rail topology."
            )
        except Exception:
            pass

        state = pclient.RecieveSimulationStandardData()
        print(f"[topology-audit] simulation state v={state}")
        if state not in EXPECTED_SIMULATION_STATES:
            pending = pclient.PeekPending(64)
            hex_bytes = " ".join(f"{value:02X}" for value in pending[:32])
            raise RuntimeError(
                f"[DESYNC] unexpected simulation state v={state}; "
                f"pending={len(pending)}B: {hex_bytes}"
            )

        # RecieveSimulationStandardData() performs the full second
        # RecieveInitializeMessage/Message2 sequence internally when v == 2.
        if state in (1, 2):
            continue

        # Ignoring the payload belonging to an active-data command would break
        # TCP framing. An audit client must see initialization before these.
        raise RuntimeError(
            "simulator requested active-data handling before rail topology "
            f"initialization completed: v={state}"
        )

    return pclient


def run_topology_audit(
    *,
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    output_dir = Path(output_dir).resolve()
    audit_path = output_dir / "topology_neighbor_audit.json"
    cache_path = output_dir / "contextual_topology_cache.npz"

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, int(port)))
        server_socket.listen(1)
        print("[topology-audit] waiting for simulator")
        print(f"[topology-audit] listen={host}:{int(port)}")

        connection, client_address = server_socket.accept()
        with connection:
            print(f"[topology-audit] simulator connected: {client_address}")
            pclient = PClient.PClient(connection)
            wait_for_initialized_topology(pclient)
            declared_count = int(pclient.RAILINE_COUNT)
            loaded_count = len(pclient.RAILLINE_DIC)
            print(
                "[topology-audit] initialization received: "
                f"declared_rail_count={declared_count}, "
                f"loaded_rail_count={loaded_count}"
            )

            try:
                topology = build_contextual_topology(
                    pclient.RAILLINE_DIC,
                    expected_rail_count=declared_count,
                    audit_path=audit_path,
                    cache_path=cache_path,
                )
            except TopologyAuditError as error:
                print(f"[topology-audit] FAILED: {error}")
                print(f"[topology-audit] failure audit: {audit_path}")
                print("[topology-audit] training was not started")
                return 2

            audit = topology.audit
            print("[topology-audit] PASSED")
            print(
                "[topology-audit] "
                f"physical_rail_count={audit['physical_rail_count']}, "
                f"controlled_rail_count={audit['controlled_rail_count']}, "
                f"boundary_rail_count={audit['boundary_rail_count']}, "
                f"boundary_rail_ids={audit['boundary_rail_ids']}"
            )
            print(
                "[topology-audit] controlled centers: "
                f"min_incoming={audit['controlled_reachable_incoming']['min']}, "
                f"min_outgoing={audit['controlled_reachable_outgoing']['min']}"
            )
            print(f"[topology-audit] topology_hash={topology.topology_hash}")
            print(f"[topology-audit] mapping_hash={topology.mapping_hash}")
            print(f"[topology-audit] audit: {audit_path}")
            print(f"[topology-audit] cache: {cache_path}")
            print("[topology-audit] training was not started")
            return 0
    finally:
        server_socket.close()


def main():
    try:
        sys.stdout.reconfigure(
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
        sys.stderr.reconfigure(
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
    except (AttributeError, ValueError):
        pass

    args = parse_args()
    return run_topology_audit(
        host=args.host,
        port=args.port,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
