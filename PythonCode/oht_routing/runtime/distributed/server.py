"""Multi-port simulator supervision for the central learner runtime."""

from __future__ import annotations

import socket
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch

from oht_routing.runtime.client import ContextualTrainingFailure
from oht_routing.runtime.server import (
    HOST,
    _configure_listener_socket,
    _run_session,
)
from oht_routing.runtime.summary import print_runtime_summary

from .coordinator import CentralTrainingCoordinator
from .worker import DistributedCollectorClient


class DistributedServerGroup:
    """Pre-bind one listener per worker and supervise reconnect loops."""

    def __init__(self, ports, clients, coordinator):
        self.ports = tuple(int(port) for port in ports)
        self.clients = tuple(clients)
        self.coordinator = coordinator
        if len(self.ports) != len(self.clients):
            raise ValueError("ports and clients must have equal lengths")
        self._listeners: list[socket.socket] = []
        self._threads: list[threading.Thread] = []
        self._active_lock = threading.Lock()
        self._active_connections: dict[int, socket.socket] = {}

    def _bind_all(self) -> None:
        listeners = []
        current = None
        try:
            for port in self.ports:
                current = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                _configure_listener_socket(current)
                current.bind((HOST, port))
                current.listen(1)
                current.settimeout(0.5)
                listeners.append(current)
                current = None
        except Exception:
            if current is not None:
                current.close()
            for server in listeners:
                server.close()
            raise
        self._listeners = listeners

    def start(self) -> None:
        self._bind_all()
        try:
            for worker_id, (server, port, client) in enumerate(zip(
                self._listeners, self.ports, self.clients
            )):
                thread = threading.Thread(
                    target=self._accept_loop,
                    args=(worker_id, server, port, client),
                    name=f"contextual-collector-{worker_id}",
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)
        except Exception:
            self.stop()
            self.join()
            raise
        print(
            "[distributed] listeners ready: "
            + ", ".join(f"{HOST}:{port}" for port in self.ports),
            flush=True,
        )

    def _accept_loop(self, worker_id, server, port, client) -> None:
        while not self.coordinator.stop_event.is_set():
            print(datetime.now().strftime("%Y.%m.%d - %H:%M:%S"))
            print(
                f"[contextual-runtime] worker={worker_id} waiting on "
                f"{HOST}:{port}"
            )
            try:
                connection, address = server.accept()
            except socket.timeout:
                continue
            except OSError:
                if self.coordinator.stop_event.is_set():
                    return
                traceback.print_exc()
                self.coordinator.report_worker_failure(
                    worker_id,
                    RuntimeError(
                        f"listener accept failed on {HOST}:{port}"
                    ),
                )
                return
            with self._active_lock:
                if self.coordinator.stop_event.is_set():
                    self._close_socket(connection)
                    return
                self._active_connections[worker_id] = connection
            try:
                with connection:
                    _run_session(connection, address, port, client)
            except ContextualTrainingFailure as error:
                self.coordinator.report_worker_failure(worker_id, error)
                return
            except (ConnectionError, IndexError, OSError):
                if not self.coordinator.stop_event.is_set():
                    traceback.print_exc()
                    print(
                        "[contextual-runtime] "
                        f"worker={worker_id} disconnected; waiting for reconnect"
                    )
            except Exception as error:
                if self.coordinator.stop_event.is_set():
                    return
                traceback.print_exc()
                if client.training_failed:
                    self.coordinator.report_worker_failure(worker_id, error)
                    return
                print(
                    "[contextual-runtime] "
                    f"worker={worker_id} session failed; waiting for reconnect"
                )
            finally:
                with self._active_lock:
                    self._active_connections.pop(worker_id, None)

    @staticmethod
    def _close_socket(target: socket.socket) -> None:
        try:
            target.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            target.close()
        except OSError:
            pass

    def stop(self) -> None:
        self.coordinator.request_stop()
        for server in self._listeners:
            self._close_socket(server)
        with self._active_lock:
            active = tuple(self._active_connections.values())
        for connection in active:
            self._close_socket(connection)

    def join(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + max(0.0, float(timeout))
        for thread in self._threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        alive = [thread.name for thread in self._threads if thread.is_alive()]
        if alive:
            print(
                "[distributed] collector threads did not stop before "
                f"timeout: {alive}",
                flush=True,
            )


def _summary_view(config, worker, coordinator):
    return SimpleNamespace(
        config=config,
        encoder=worker.encoder,
        rail_tat_diagnostic_path=(
            Path(config.rail_tat_diagnostic_path)
            if config.rail_tat_diagnostic_path else None
        ),
        checkpoint_loaded=False,
        environment_capture=None,
        device=torch.device(config.device),
    )


def serve_distributed(config) -> None:
    """Launch collectors and run the sole CUDA owner on the main thread."""

    ports = tuple(config.sim_ports or ())
    coordinator = CentralTrainingCoordinator(config)
    servers = None
    status = "failed"
    try:
        workers = tuple(
            DistributedCollectorClient(config, worker_id, coordinator)
            for worker_id in range(config.num_sim)
        )
        print_runtime_summary(_summary_view(config, workers[0], coordinator))
        print(
            "[distributed] "
            f"num_sim={config.num_sim}, ports={ports}, "
            "collector_device=cpu, central_device="
            f"{coordinator.device}, microbatch_window_ms="
            f"{coordinator.microbatch_window_s * 1000.0:g}",
            flush=True,
        )
        servers = DistributedServerGroup(ports, workers, coordinator)
        servers.start()
        status = "stopped"
        coordinator.run()
    except KeyboardInterrupt:
        status = "interrupted"
        coordinator.request_stop()
    except Exception:
        status = "failed"
        coordinator.request_stop()
        raise
    finally:
        if servers is not None:
            servers.stop()
            servers.join()
        coordinator.close(status=status)


__all__ = (
    "DistributedServerGroup",
    "serve_distributed",
)
