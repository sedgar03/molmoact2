from typing import Dict

import numpy as np

from gello_min.robot import Robot
from i2rt.robots.utils import GripperType


class YAMRobot(Robot):
    """A class representing a simulated YAM robot."""

    def __init__(
        self,
        channel="can0",
        zero_gravity_mode=False,
        limit_gripper_force=20.0,
    ):
        from i2rt.robots.get_robot import get_yam_robot

        self.zero_gravity_mode = bool(zero_gravity_mode)
        self.limit_gripper_force = float(limit_gripper_force)
        self.robot = get_yam_robot(
            channel=channel,
            gripper_type=GripperType.LINEAR_4310,
            zero_gravity_mode=self.zero_gravity_mode,
            limit_gripper_force=self.limit_gripper_force,
        )

        # YAM has 7 joints (6 arm joints + 1 gripper)
        self._joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "gripper",
        ]
        self._joint_state = self.get_joint_state()  # robot stays where it was when reboot
        # self._joint_state = np.zeros(7)  # robot goes immediately to reset position (avoid using)
        self._joint_velocities = np.zeros(7)  # 7 joints
        self._joint_efforts = np.zeros(7)  # 7 joints
        self._gripper_state = 0.0  # didn't use because joint_state includes gripper position

    def num_dofs(self) -> int:
        return 7  # YAM has 7 DOFs

    def get_joint_state(self) -> np.ndarray:
        # Get actual joint positions from I2RT robot (7 joints total)
        joint_pos = self.robot.get_joint_pos()
        self._joint_state = self._as_7d(joint_pos)
        return self._joint_state

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert (
            len(joint_state) == self.num_dofs()
        ), f"Expected {self.num_dofs()} joint values, got {len(joint_state)}"

        dt = 0.01
        self._joint_velocities = (joint_state - self._joint_state) / dt
        self._joint_state = joint_state

        # Command the I2RT robot with all 7 joints (6 arm + 1 gripper)
        self.command_joint_pos(joint_state)

    def get_observations(self) -> Dict[str, np.ndarray]:
        robot_obs = self.robot.get_observations()
        joint_pos = self._combine_arm_gripper(
            robot_obs.get("joint_pos", robot_obs.get("joint_positions")),
            robot_obs.get("gripper_pos", robot_obs.get("gripper_position")),
        )
        joint_vel = self._combine_arm_gripper(
            robot_obs.get("joint_vel", robot_obs.get("joint_velocities")),
            robot_obs.get("gripper_vel", robot_obs.get("gripper_velocity")),
        )
        joint_eff = self._combine_arm_gripper(
            robot_obs.get("joint_eff", robot_obs.get("joint_efforts")),
            robot_obs.get("gripper_eff", robot_obs.get("gripper_effort")),
        )
        if joint_pos is None:
            joint_pos = self.get_joint_state()
        else:
            self._joint_state = joint_pos
        if joint_vel is None:
            joint_vel = self._joint_velocities
        else:
            self._joint_velocities = joint_vel
        if joint_eff is None:
            joint_eff = self._joint_efforts
        else:
            self._joint_efforts = joint_eff

        ee_pos_quat = np.zeros(7)  # Placeholder for FK
        return {
            "joint_positions": joint_pos,
            "joint_velocities": joint_vel,
            "joint_efforts": joint_eff,
            "joint_eff": joint_eff,
            "ee_pos_quat": ee_pos_quat,
            "gripper_position": np.array([joint_pos[-1]]),
            "gripper_velocity": np.array([joint_vel[-1]]),
            "gripper_effort": np.array([joint_eff[-1]]),
            "gripper_eff": np.array([joint_eff[-1]]),
            "robot_config": {
                "zero_gravity_mode": self.zero_gravity_mode,
                "limit_gripper_force": self.limit_gripper_force,
            },
        }

    def get_joint_pos(self):
        # Get 7 joints from I2RT robot (6 arm + 1 gripper)
        joint_pos = self.robot.get_joint_pos()
        return self._as_7d(joint_pos)

    def command_joint_pos(self, target_pos):
        # Ensure we send exactly 7 joints to the I2RT robot
        if len(target_pos) > 7:
            target_pos = target_pos[:7]
        elif len(target_pos) < 7:
            # Pad with zeros if we have fewer than 7 joints
            target_pos = np.pad(target_pos, (0, 7 - len(target_pos)), "constant")
        self.robot.command_joint_pos(np.array(target_pos))

    @staticmethod
    def _as_7d(values) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1)
        if len(values) > 7:
            return values[:7]
        if len(values) < 7:
            return np.pad(values, (0, 7 - len(values)), "constant")
        return values

    @classmethod
    def _combine_arm_gripper(cls, arm_values, gripper_values) -> np.ndarray | None:
        if arm_values is None:
            return None
        arm = np.asarray(arm_values, dtype=float).reshape(-1)
        if len(arm) >= 7:
            return cls._as_7d(arm)
        if gripper_values is None:
            return cls._as_7d(arm)
        gripper = np.asarray(gripper_values, dtype=float).reshape(-1)
        gripper_value = gripper[0] if len(gripper) else 0.0
        return cls._as_7d(np.concatenate([arm, [gripper_value]]))


def main():
    robot = YAMRobot()
    print(robot.get_observations())


if __name__ == "__main__":
    main()
