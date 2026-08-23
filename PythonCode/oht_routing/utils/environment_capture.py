"""Streaming actor-inference environment capture for bottleneck analysis."""

from __future__ import annotations

import gzip
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from simulator.oht import OHTState

from oht_routing.version import CONTEXTUAL_VERSION


DEFAULT_CAPTURE_STEPS = 2_000
CAPTURE_SCHEMA = "actor_environment_capture_v1"


def _integer(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _integer_list(values):
    return [_integer(value) for value in (values or [])]


def _pass_time_record(item):
    return {
        "rail_id": _integer(getattr(item, "ID", -1), -1),
        "state": _integer(getattr(item, "State", -1), -1),
        "time_s": _number(getattr(item, "PassTime", None)),
        "distance_mm": _number(getattr(item, "Distance", None)),
    }


def _command_time_records(values):
    records = []
    for command_id, item in sorted((values or {}).items(), key=lambda pair: int(pair[0])):
        records.append({
            "command_id": _integer(command_id),
            "oht_tat_s": _number(getattr(item, "OHTTat", None)),
            "command_tat_s": _number(getattr(item, "CmdTat", None)),
            "oht_work_ratio": _number(
                getattr(item, "OHTWorkTimeByCommand", None)
            ),
        })
    return records


def _state_name(state):
    try:
        return OHTState(int(state)).name
    except (TypeError, ValueError):
        return "UNKNOWN"


def _diagnostic_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    return _number(value)


class ActorEnvironmentCapture:
    """Write the first actor-evaluation ticks as recoverable gzip JSONL."""

    def __init__(
        self,
        output_root: str | Path,
        *,
        max_steps: int = DEFAULT_CAPTURE_STEPS,
        flush_interval: int = 25,
        timestamp: str | None = None,
    ):
        self.max_steps = int(max_steps)
        self.flush_interval = int(flush_interval)
        if self.max_steps <= 0:
            raise ValueError("environment capture max_steps must be positive")
        if self.flush_interval <= 0:
            raise ValueError("environment capture flush_interval must be positive")

        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"capture_{stamp}_{CONTEXTUAL_VERSION}_actor_inference"
        self.output_dir = self._allocate_directory(root, base_name)
        self.manifest_path = self.output_dir / "manifest.json"
        self.topology_path = self.output_dir / "rail_topology.json.gz"
        self.steps_path = self.output_dir / "steps.jsonl.gz"
        self.record_count = 0
        self._stream = None
        self._closed = False
        self._topology_written = False
        self._dwell_state = {}
        self._last_sim_time = None
        self._manifest = {
            "schema": CAPTURE_SCHEMA,
            "version": CONTEXTUAL_VERSION,
            "mode": "actor_inference",
            "status": "initialized",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "max_steps": self.max_steps,
            "record_count": 0,
            "files": {
                "manifest": self.manifest_path.name,
                "rail_topology": self.topology_path.name,
                "steps": self.steps_path.name,
            },
        }
        self._write_manifest()

    @staticmethod
    def _allocate_directory(root: Path, base_name: str) -> Path:
        for index in range(10_000):
            suffix = "" if index == 0 else f"_{index:02d}"
            candidate = root / f"{base_name}{suffix}"
            try:
                candidate.mkdir(exist_ok=False)
                return candidate
            except FileExistsError:
                continue
        raise RuntimeError("could not allocate a unique environment capture directory")

    @property
    def completed(self) -> bool:
        return self.record_count >= self.max_steps

    def _write_manifest(self):
        temporary = self.manifest_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self._manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(self.manifest_path)

    def _write_topology(self, pclient, client):
        controlled = set(
            _integer(value)
            for value in getattr(
                getattr(client, "topology", None),
                "controlled_rail_ids",
                (),
            )
        )
        rails = []
        for rail_id, rail in sorted(
            getattr(pclient, "RAILLINE_DIC", {}).items(),
            key=lambda pair: int(pair[0]),
        ):
            distance = _number(getattr(rail, "Distance", None))
            free_flow_time = _number(
                getattr(rail, "DistancePerVelocity", None)
            )
            nominal_velocity = (
                distance / free_flow_time
                if distance is not None
                and free_flow_time is not None
                and free_flow_time > 0.0
                else None
            )
            rails.append({
                "rail_id": _integer(rail_id),
                "sim_id": _integer(getattr(rail, "SimID", 0)),
                "controlled": _integer(rail_id) in controlled,
                "distance_mm": distance,
                "free_flow_time_s": free_flow_time,
                "nominal_velocity_mm_s": nominal_velocity,
                "line_type": _integer(getattr(rail, "LineType", 0)),
                "port_count": _integer(getattr(rail, "PortCount", 0)),
                "incoming_rail_ids": _integer_list(
                    getattr(rail, "LevelJoiningLineIDList", [])
                ),
                "incoming_level2_rail_ids": _integer_list(
                    getattr(rail, "Level2JoiningLineIDList", [])
                ),
                "incoming_level3_rail_ids": _integer_list(
                    getattr(rail, "Level3JoiningLineIDList", [])
                ),
                "outgoing_rail_ids": _integer_list(
                    getattr(rail, "DivergingLineIDList", [])
                ),
            })
        topology = getattr(client, "topology", None)
        payload = {
            "schema": CAPTURE_SCHEMA,
            "version": CONTEXTUAL_VERSION,
            "topology_hash": getattr(topology, "topology_hash", None),
            "mapping_hash": getattr(topology, "mapping_hash", None),
            "rail_count": len(rails),
            "rails": rails,
        }
        with gzip.open(
            self.topology_path,
            "wt",
            encoding="utf-8",
            compresslevel=1,
        ) as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("\n")
        self._topology_written = True
        self._manifest.update({
            "topology_hash": payload["topology_hash"],
            "mapping_hash": payload["mapping_hash"],
            "rail_count": len(rails),
        })

    @staticmethod
    def _vector_by_rail(rail_ids, values):
        if values is None:
            return {}
        try:
            return {
                _integer(rail_id): _number(values[index])
                for index, rail_id in enumerate(rail_ids)
            }
        except (IndexError, TypeError):
            return {}

    def _rail_records(self, pclient, client):
        topology = getattr(client, "topology", None)
        all_rail_ids = getattr(topology, "all_rail_ids", ())
        controlled_rail_ids = getattr(topology, "controlled_rail_ids", ())
        baseline = self._vector_by_rail(
            all_rail_ids, getattr(client, "_current_baseline", None)
        )
        policy = self._vector_by_rail(
            controlled_rail_ids, getattr(client, "last_policy_action", None)
        )
        applied = self._vector_by_rail(
            controlled_rail_ids, getattr(client, "last_applied_action", None)
        )
        costs = getattr(pclient, "RAILLINECOST_DIC", {})
        records = []
        occupancy = {}
        unknown_oht_ids = set()
        known_oht_ids = {
            _integer(value) for value in getattr(pclient, "OHT_DIC", {})
        }
        for rail_id, rail in sorted(
            getattr(pclient, "RAILLINE_DIC", {}).items(),
            key=lambda pair: int(pair[0]),
        ):
            rail_id = _integer(rail_id)
            oht_ids = _integer_list(getattr(rail, "OhtList", []))
            for oht_id in oht_ids:
                occupancy.setdefault(oht_id, []).append(rail_id)
                if oht_id not in known_oht_ids:
                    unknown_oht_ids.add(oht_id)
            cost = costs.get(rail_id)
            records.append({
                "rail_id": rail_id,
                "oht_ids": oht_ids,
                "occupancy_count": len(oht_ids),
                "predicted_oht_ids": _integer_list(
                    getattr(rail, "PredictedOHTIDList", [])
                ),
                "predicted_oht_count": _integer(
                    getattr(rail, "PredictedOHTCount", 0)
                ),
                "sim_average_speed": _number(
                    getattr(rail, "SimAvgSpeed", None)
                ),
                "idle_oht_count": _integer(
                    getattr(rail, "IdleOHTCount", 0)
                ),
                "reservation_port_count": _integer(
                    getattr(rail, "ReservationPortCount", 0)
                ),
                "baseline_cost": baseline.get(rail_id),
                "sent_cost": _number(getattr(cost, "FRailLineCost", None)),
                "policy_action": policy.get(rail_id),
                "applied_action": applied.get(rail_id),
                "parameter_dw": _number(
                    getattr(client, "parameterDw", {}).get(rail_id)
                ),
                "parameter_c": _number(
                    getattr(client, "parameterC", {}).get(rail_id)
                ),
            })
        return records, occupancy, sorted(unknown_oht_ids)

    def _update_dwell(self, oht_id, rail_id, sim_time):
        if rail_id is None or sim_time is None:
            self._dwell_state.pop(oht_id, None)
            return None, True
        previous = self._dwell_state.get(oht_id)
        if previous is None or previous[0] != rail_id:
            lower_bound = previous is None
            self._dwell_state[oht_id] = (rail_id, sim_time, lower_bound)
            return 0.0, lower_bound
        entered_at = previous[1]
        return max(0.0, sim_time - entered_at), bool(previous[2])

    def _oht_records(self, pclient, occupancy, sim_time):
        records = []
        missing_from_rails = []
        multiple_rail_membership = []
        observed_oht_ids = set()
        for oht_id, oht in sorted(
            getattr(pclient, "OHT_DIC", {}).items(),
            key=lambda pair: int(pair[0]),
        ):
            oht_id = _integer(oht_id)
            observed_oht_ids.add(oht_id)
            occupancy_rails = list(occupancy.get(oht_id, []))
            not_pass_times = list(getattr(oht, "NotPassTimes", None) or [])
            reported_current = (
                _integer(getattr(not_pass_times[0], "ID", -1), -1)
                if not_pass_times else None
            )
            current_rail = (
                reported_current
                if reported_current is not None and reported_current >= 0
                else occupancy_rails[0]
                if len(occupancy_rails) == 1
                else None
            )
            if not occupancy_rails:
                missing_from_rails.append(oht_id)
            if len(occupancy_rails) > 1:
                multiple_rail_membership.append(oht_id)
            observed_dwell, lower_bound = self._update_dwell(
                oht_id, current_rail, sim_time
            )
            current_rail_object = getattr(pclient, "RAILLINE_DIC", {}).get(
                current_rail
            )
            current_distance = _number(getattr(oht, "CurrentDistance", None))
            rail_distance = _number(
                getattr(current_rail_object, "Distance", None)
            )
            progress = (
                current_distance / rail_distance
                if current_distance is not None
                and rail_distance is not None
                and rail_distance > 0.0
                else None
            )
            front_ohts = {
                str(_integer(rail_id)): [
                    {
                        "oht_id": _integer(getattr(item, "ID", -1), -1),
                        "distance_mm": _number(
                            getattr(item, "Distance", None)
                        ),
                    }
                    for item in (items or [])
                ]
                for rail_id, items in sorted(
                    (getattr(oht, "FrontOhts", {}) or {}).items(),
                    key=lambda pair: int(pair[0]),
                )
            }
            vel_by_line = {
                str(_integer(rail_id)): _number(velocity)
                for rail_id, velocity in sorted(
                    (getattr(oht, "VelByLine", {}) or {}).items(),
                    key=lambda pair: int(pair[0]),
                )
            }
            state = _integer(getattr(oht, "State", -1), -1)
            records.append({
                "oht_id": oht_id,
                "name": str(getattr(oht, "Name", "") or ""),
                "state": state,
                "state_name": _state_name(state),
                "current_rail_id": current_rail,
                "occupancy_rail_ids": occupancy_rails,
                "reported_current_rail_id": reported_current,
                "reported_current_rail_time_s": (
                    _number(getattr(not_pass_times[0], "PassTime", None))
                    if not_pass_times else None
                ),
                "observed_current_rail_dwell_s": observed_dwell,
                "observed_dwell_is_lower_bound": lower_bound,
                "current_distance_mm": current_distance,
                "current_rail_progress_ratio": progress,
                "job_id": _integer(getattr(oht, "JobID", 0)),
                "destination_rail_id": _integer(
                    getattr(oht, "DestinationLine", -1), -1
                ),
                "route_rail_ids": _integer_list(
                    getattr(oht, "RouteList", [])
                ),
                "idle_time_s": _number(getattr(oht, "IdleTime", None)),
                "stop_time_s": _number(getattr(oht, "StopTime", None)),
                "remaining_time_s": _number(
                    getattr(oht, "RemainTime", None)
                ),
                "remaining_distance_mm": _number(
                    getattr(oht, "RemainingDistanace", None)
                ),
                "operation_rate": _number(
                    getattr(oht, "OperationRate", None)
                ),
                "operation_tat_s": _number(
                    getattr(oht, "OperationTAT", None)
                ),
                "individual_tat_s": _number(
                    getattr(oht, "OhtIndividualTat", None)
                ),
                "dispatched_command_id": _integer(
                    getattr(oht, "DispatchedCommand", 0)
                ),
                "running_area_type": _integer(
                    getattr(oht, "RunningAreaType", -1), -1
                ),
                "carrier_types": _integer_list(
                    getattr(oht, "CarrierTypes", [])
                ),
                "pass_distance_mm": _number(
                    getattr(oht, "PassDistance", None)
                ),
                "velocity_by_rail": vel_by_line,
                "front_ohts_by_rail": front_ohts,
                "passed_rails": [
                    _pass_time_record(item)
                    for item in (getattr(oht, "PassTimes", None) or [])
                ],
                "current_rail_reports": [
                    _pass_time_record(item) for item in not_pass_times
                ],
                "command_tat": _command_time_records(
                    getattr(oht, "CmdCompleteTat", {})
                ),
            })
        for stale_oht_id in set(self._dwell_state).difference(observed_oht_ids):
            self._dwell_state.pop(stale_oht_id, None)
        return records, missing_from_rails, multiple_rail_membership

    @staticmethod
    def _job_records(pclient):
        records = []
        for job_id, job in sorted(
            getattr(pclient, "JOB_DIC", {}).items(),
            key=lambda pair: int(pair[0]),
        ):
            records.append({
                "job_id": _integer(job_id),
                "state": _integer(getattr(job, "State", -1), -1),
                "priority": _integer(getattr(job, "Priority", -1), -1),
                "oht_id": _integer(getattr(job, "OHTId", 0)),
                "from_node_id": _integer(
                    getattr(job, "FromNode", -1), -1
                ),
                "to_node_id": _integer(getattr(job, "ToNode", -1), -1),
                "is_equipment": bool(getattr(job, "IsEqp", 0)),
                "reassign_count": _integer(
                    getattr(job, "ReAssignCount", 0)
                ),
                "route_rail_ids": _integer_list(
                    getattr(job, "RouteList", [])
                ),
                "waiting_passed_rails": [
                    _pass_time_record(item)
                    for item in (getattr(job, "Waiting_PassLines", None) or [])
                ],
                "transfer_passed_rails": [
                    _pass_time_record(item)
                    for item in (getattr(job, "Transfer_PassLines", None) or [])
                ],
                "carrier_types": _integer_list(
                    getattr(job, "CarrierTypes", [])
                ),
                "running_area_types": _integer_list(
                    getattr(job, "RunningAreaTyes", [])
                ),
            })
        return records

    @staticmethod
    def _diagnostics(client):
        prefixes = (
            "env/",
            "episode/",
            "action/",
            "curriculum/",
            "b_rl/",
            "cost/",
            "reward/",
            "local/",
            "oht/",
            "termination/",
            "warmup/",
            "runtime/",
            "lead/",
            "leadlag/",
        )
        return {
            key: _diagnostic_value(value)
            for key, value in getattr(client, "last_diagnostics", {}).items()
            if key.startswith(prefixes)
        }

    def capture(self, pclient, client) -> bool:
        if self._closed or self.completed:
            return False
        if not self._topology_written:
            self._write_topology(pclient, client)
        if self._stream is None:
            self._stream = gzip.open(
                self.steps_path,
                "wt",
                encoding="utf-8",
                compresslevel=1,
            )

        sim_time = _number(getattr(pclient, "SimTime", None))
        if (
            self._last_sim_time is not None
            and sim_time is not None
            and sim_time < self._last_sim_time
        ):
            self._dwell_state.clear()
        self._last_sim_time = sim_time

        rails, occupancy, unknown_oht_ids = self._rail_records(pclient, client)
        ohts, missing_from_rails, multiple_rail_membership = self._oht_records(
            pclient, occupancy, sim_time
        )
        jobs = self._job_records(pclient)
        diagnostics = self._diagnostics(client)
        record = {
            "schema": CAPTURE_SCHEMA,
            "version": CONTEXTUAL_VERSION,
            "capture_index": self.record_count,
            "wall_time_utc": datetime.now(timezone.utc).isoformat(),
            "global_step": _integer(
                diagnostics.get("env/step", getattr(client, "total_steps", 0))
            ),
            "episode_id": _integer(
                diagnostics.get("env/episode", getattr(client, "episode_id", 0))
            ),
            "episode_step": _integer(
                diagnostics.get(
                    "episode/step", getattr(client, "episode_steps", 0)
                )
            ),
            "sim_time_s": sim_time,
            "environment": {
                "total_tat_s": _number(getattr(pclient, "TotalTat", None)),
                "recent_completed_tat_300s_mean_s": _number(
                    getattr(pclient, "RecentCompletedTat300s", None)
                ),
                "recent_completed_tat_300s_p90_s": _number(
                    getattr(pclient, "RecentCompletedTat300sP90", None)
                ),
                "recent_completed_tat_300s_count": _integer(
                    getattr(pclient, "RecentCompletedTat300sCount", 0)
                ),
                "recent_completed_tat_300s_available": bool(
                    getattr(pclient, "RecentCompletedTat300sAvailable", False)
                ),
                "operation_rate": _number(
                    getattr(pclient, "TotalOhtOperationRate", None)
                ),
                "completed_command_count": _integer(
                    getattr(pclient, "CompletedCommandCount", 0)
                ),
                "transferring_command_count": _integer(
                    getattr(pclient, "TransferCommandCount", 0)
                ),
                "waiting_command_count": _integer(
                    getattr(pclient, "WaitingCommandCount", 0)
                ),
                "queued_command_count": _integer(
                    getattr(pclient, "QueuedCommandCount", 0)
                ),
                "rail_count": len(rails),
                "oht_count": len(ohts),
                "job_count": len(jobs),
            },
            "consistency": {
                "unknown_oht_ids_on_rails": unknown_oht_ids,
                "oht_ids_missing_from_rail_occupancy": missing_from_rails,
                "oht_ids_on_multiple_rails": multiple_rail_membership,
            },
            "diagnostics": diagnostics,
            "rails": rails,
            "ohts": ohts,
            "jobs": jobs,
        }
        if self.record_count == 0:
            self._manifest.update({
                "status": "capturing",
                "start_global_step": record["global_step"],
                "start_episode_id": record["episode_id"],
                "start_episode_step": record["episode_step"],
                "start_sim_time_s": record["sim_time_s"],
            })
        self._manifest.update({
            "last_global_step": record["global_step"],
            "last_episode_id": record["episode_id"],
            "last_episode_step": record["episode_step"],
            "last_sim_time_s": record["sim_time_s"],
        })
        self._stream.write(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        )
        self._stream.write("\n")
        self.record_count += 1

        if self.record_count % self.flush_interval == 0:
            self._stream.flush()
            self._manifest["record_count"] = self.record_count
            self._manifest["status"] = "capturing"
            self._write_manifest()
        if self.completed:
            self.close(status="completed")
        return True

    def close(self, *, status="stopped"):
        if self._closed:
            return
        if self._stream is not None:
            self._stream.flush()
            self._stream.close()
            self._stream = None
        self._closed = True
        self._manifest.update({
            "status": "completed" if self.completed else str(status),
            "record_count": self.record_count,
            "closed_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        self._manifest["file_sizes_bytes"] = {
            "rail_topology": (
                self.topology_path.stat().st_size
                if self.topology_path.exists()
                else 0
            ),
            "steps": (
                self.steps_path.stat().st_size if self.steps_path.exists() else 0
            ),
        }
        self._write_manifest()


__all__ = (
    "ActorEnvironmentCapture",
    "CAPTURE_SCHEMA",
    "DEFAULT_CAPTURE_STEPS",
)
