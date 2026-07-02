"""Audit YAM effort-unit provenance from the installed i2rt configuration.

This is a no-motion software audit. It does not open CAN or command the robot.
It reports the motor types, configured direction signs, and DM torque feedback
ranges used to decode ``joint_eff``/``joint_efforts``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import tyro


@dataclass
class Args:
    arm: str = "yam"
    """YAM arm type: yam, yam_pro, yam_ultra, or big_yam."""

    gripper: str = "linear_4310"
    """Follower gripper type used for the run."""

    output_path: Optional[str] = None
    """Optional JSON output path."""


def _constants_dict(constants: Any) -> dict[str, float]:
    return {
        "position_min": float(constants.POSITION_MIN),
        "position_max": float(constants.POSITION_MAX),
        "velocity_min": float(constants.VELOCITY_MIN),
        "velocity_max": float(constants.VELOCITY_MAX),
        "torque_min_nm": float(constants.TORQUE_MIN),
        "torque_max_nm": float(constants.TORQUE_MAX),
    }


def main() -> None:
    args = tyro.cli(Args)

    from i2rt.motor_drivers.utils import MotorType
    from i2rt.robots.utils import ArmType, GripperType, _load_arm_config

    arm_type = ArmType.from_string_name(args.arm)
    gripper_type = GripperType.from_string_name(args.gripper)
    arm_cfg = _load_arm_config(arm_type)

    joints: list[dict[str, Any]] = []
    for idx, ((can_id, motor_type), direction) in enumerate(
        zip(arm_cfg.motor_list, arm_cfg.directions, strict=True)
    ):
        joints.append(
            {
                "joint_index": idx,
                "can_id": int(can_id),
                "motor_type": str(motor_type),
                "direction": float(direction),
                "constants": _constants_dict(MotorType.get_motor_constants(motor_type)),
            }
        )

    if gripper_type not in (GripperType.NO_GRIPPER, GripperType.YAM_TEACHING_HANDLE):
        motor_type = gripper_type.get_motor_type(arm_type)
        joints.append(
            {
                "joint_index": len(joints),
                "can_id": 7,
                "motor_type": str(motor_type),
                "direction": float(gripper_type.get_motor_direction(arm_type)),
                "constants": _constants_dict(MotorType.get_motor_constants(motor_type)),
                "is_gripper": True,
            }
        )

    report = {
        "schema": "yam_effort_unit_audit_v1",
        "arm": args.arm,
        "gripper": args.gripper,
        "joint_count": len(joints),
        "joint_effort_source": (
            "DM feedback frame torque decoded by i2rt.motor_drivers.dm_driver."
            "parse_recv_message using MotorType TORQUE_MIN/TORQUE_MAX, then stored"
            " as MotorInfo.eff = state.torque * motor_direction."
        ),
        "effort_unit_interpretation": (
            "Nominal Nm in DM motor torque-frame units. Absolute physical calibration"
            " still depends on motor firmware/model constants and drivetrain assumptions."
        ),
        "joints": joints,
    }

    text = json.dumps(report, indent=2) + "\n"
    print(text)
    if args.output_path:
        path = Path(args.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
