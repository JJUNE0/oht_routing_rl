"""Fixed simulator-stage contracts for contextual TD7."""

from __future__ import annotations


STAGE_ONE = 1
STAGE_TWO = 2
STAGE_ONE_SIM_END_TIME = 2_000
STAGE_TWO_SIM_END_TIME = 45_000
STAGE_TWO_STAGE1_POLICY_STEPS = STAGE_ONE_SIM_END_TIME
STAGE_SIM_END_TIMES = {
    STAGE_ONE: STAGE_ONE_SIM_END_TIME,
    STAGE_TWO: STAGE_TWO_SIM_END_TIME,
}
VALID_STAGES = tuple(STAGE_SIM_END_TIMES)


def sim_end_time_for_stage(stage: int) -> int:
    try:
        return STAGE_SIM_END_TIMES[int(stage)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"stage must be one of {VALID_STAGES}") from error


__all__ = (
    "STAGE_ONE",
    "STAGE_ONE_SIM_END_TIME",
    "STAGE_SIM_END_TIMES",
    "STAGE_TWO",
    "STAGE_TWO_SIM_END_TIME",
    "STAGE_TWO_STAGE1_POLICY_STEPS",
    "VALID_STAGES",
    "sim_end_time_for_stage",
)
