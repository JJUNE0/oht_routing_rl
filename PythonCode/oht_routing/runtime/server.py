"""TCP server lifecycle for the contextual simulator protocol."""

import socket
import os
from pathlib import Path
import traceback
from datetime import datetime

from simulator import client as PClient
from oht_routing.runtime.client import ContextualTrainingFailure
from oht_routing.runtime.protocol import handle_command, read_port


HOST = "127.0.0.1"
EXPECTED_SIMULATION_STATES = {0, 1, 2, 3, 4, 5, 6}


def _configure_listener_socket(server):
    """Make listener ownership exclusive on Windows and reusable elsewhere.

    Windows gives ``SO_REUSEADDR`` different semantics from POSIX and may let
    two live processes bind the same address. That is unsafe for an explicit
    simulator-to-worker port mapping, so prefer exclusive ownership whenever
    the platform exposes it.
    """

    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        server.setsockopt(
            socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1
        )
    else:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


def _run_session(connection, address, port, client):
    print(
        "[contextual-runtime] TCP accepted; "
        f"starting PClient initialization handshake: {address}"
    )
    # PClient owns the fixed-width initialization/reset handshake.
    # Passing the override avoids appending duplicate raw bytes.
    config = client.config
    pclient = PClient.PClient(connection, sim_end_time=config.sim_end_time)
    client.on_new_connection()
    print(datetime.now().strftime("%Y.%m.%d - %H:%M:%S"))
    message = (
        "[contextual-runtime] PClient initialization completed: "
        f"peer={address}, listen={HOST}:{port}"
    )
    print(message)
    pclient.WriteAdminLog(message)
    if pclient.RAILINE_COUNT == 0:
        print(
            "[contextual-runtime] connected without a model (rails=0). "
            "In the GUI Python TCP/IP tab, use Open Files and select the "
            "single-input FOLDER. Its batch run sends the model reset; "
            "ordinary Open/Run does not. Waiting for model initialization.",
            flush=True,
        )

    command_count = 0
    while True:
        command = pclient.RecieveSimulationStandardData()
        command_count += 1
        if command != 0 or command_count % config.console_log_interval == 0:
            print(datetime.now().strftime("%Y.%m.%d - %H:%M:%S"))
            print(f"[contextual-runtime] command={command}, count={command_count}")
            pclient.WriteAdminLog("RecieveSimulationStandardData.")
        if command not in EXPECTED_SIMULATION_STATES:
            pending = pclient.PeekPending(64)
            preview = " ".join(f"{value:02X}" for value in pending[:32])
            raise RuntimeError(
                f"[DESYNC] unexpected v={command}; "
                f"pending={len(pending)}B: {preview}"
            )
        if command in (0, 3, 4, 5, 6) and pclient.RAILINE_COUNT == 0:
            raise ConnectionError(
                "[MODEL NOT LOADED] GUI sent an active frame with rails=0. "
                "Stop this GUI run and restart Python. Enable TCP/IP, then "
                "use Open Files to select the single-input FOLDER; do not "
                "use ordinary Open followed by Run. No policy step was run."
            )
        handle_command(command, pclient, client)
        if command == 1 and os.environ.get("PINOKIO_HEADLESS") == "1":
            # The headless host owns exactly one episode per connection.
            # Save terminal artifacts first, then close without an expected
            # ConnectionResetError traceback. GUI sessions still reuse TCP.
            return


def _accept_sessions(server, port, client):
    stop_file = os.environ.get("PINOKIO_STOP_FILE")
    while True:
        print(datetime.now().strftime("%Y.%m.%d - %H:%M:%S"))
        print(f"[contextual-runtime] waiting on {HOST}:{port}")
        while True:
            if stop_file and Path(stop_file).is_file():
                return
            try:
                connection, address = server.accept()
                break
            except socket.timeout:
                if not stop_file:
                    raise
        with connection:
            try:
                _run_session(connection, address, port, client)
            except ContextualTrainingFailure:
                raise
            except (ConnectionError, IndexError):
                traceback.print_exc()
                print("[contextual-runtime] disconnected; waiting for reconnect")
            except FloatingPointError:
                raise
            except Exception:
                traceback.print_exc()
                if client.training_failed:
                    raise ContextualTrainingFailure(
                        "training failure latched; operator restart required"
                    )
                print("[contextual-runtime] session failed; waiting for reconnect")


def serve_contextual(client, port=None):
    """Serve simulator sessions until interrupted or training fails."""
    port = read_port() if port is None else int(port)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _configure_listener_socket(server)
    server.bind((HOST, port))
    server.listen(1)
    if os.environ.get("PINOKIO_STOP_FILE"):
        server.settimeout(1.0)
    capture_status = "stopped"
    try:
        _accept_sessions(server, port, client)
    except ContextualTrainingFailure as error:
        capture_status = "failed"
        client.wandb_logger.finish_failed(error, client.failure_env_step)
        raise
    except KeyboardInterrupt:
        capture_status = "interrupted"
        client.wandb_logger.finish_interrupted()
        raise
    finally:
        client.close_environment_capture(status=capture_status)
        if not client.training_failed:
            client.wandb_logger.finish_success()
        server.close()
