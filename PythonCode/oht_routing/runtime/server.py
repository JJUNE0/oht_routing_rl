"""TCP server lifecycle for the contextual simulator protocol."""

import socket
import traceback
from datetime import datetime

from simulator import client as PClient
from oht_routing.runtime.client import ContextualTrainingFailure
from oht_routing.runtime.protocol import handle_command, read_port


HOST = "127.0.0.1"
EXPECTED_SIMULATION_STATES = {0, 1, 2, 3, 4, 5, 6}


def _run_session(connection, address, port, args, client):
    print(
        "[contextual-runtime] TCP accepted; "
        f"starting PClient initialization handshake: {address}"
    )
    # PClient owns the fixed-width initialization/reset handshake.
    # Passing the override avoids appending duplicate raw bytes.
    pclient = PClient.PClient(connection, sim_end_time=args.sim_end_time)
    client.on_new_connection()
    print(datetime.now().strftime("%Y.%m.%d - %H:%M:%S"))
    message = (
        "[contextual-runtime] PClient initialization completed: "
        f"peer={address}, listen={HOST}:{port}"
    )
    print(message)
    pclient.WriteAdminLog(message)

    command_count = 0
    while True:
        command = pclient.RecieveSimulationStandardData()
        command_count += 1
        if command != 0 or command_count % args.console_log_interval == 0:
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
        handle_command(
            command,
            pclient,
            client,
            sim_end_time=args.sim_end_time,
            console_log_interval=args.console_log_interval,
        )


def _accept_sessions(server, port, args, client):
    while True:
        print(datetime.now().strftime("%Y.%m.%d - %H:%M:%S"))
        print(f"[contextual-runtime] waiting on {HOST}:{port}")
        connection, address = server.accept()
        with connection:
            try:
                _run_session(connection, address, port, args, client)
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


def serve_contextual(args, client):
    """Serve simulator sessions until interrupted or training fails."""
    port = read_port()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, port))
    server.listen(1)
    try:
        _accept_sessions(server, port, args, client)
    except ContextualTrainingFailure as error:
        client.wandb_logger.finish_failed(error, client.failure_env_step)
        raise
    except KeyboardInterrupt:
        client.wandb_logger.finish_interrupted()
        raise
    finally:
        if not client.training_failed:
            client.wandb_logger.finish_success()
        server.close()
