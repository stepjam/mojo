from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from dm_control import mjcf
from mujoco_utils import mjcf_utils
from typing_extensions import Self

from mojo.elements.element import MujocoElement
from mojo.elements.joint import Joint

if TYPE_CHECKING:
    from mojo import Mojo


class Actuator(MujocoElement):
    pass


class GeneralActuator(Actuator):
    @staticmethod
    def get(
        mojo: Mojo,
        name: str,
    ) -> Self:
        mjcf = mjcf_utils.safe_find(mojo.root_element.mjcf.actuator, "general", name)
        return GeneralActuator(mojo, mjcf)

    @staticmethod
    def _create(
        tag: str,
        mojo: Mojo,
        joint: Joint,
        ctrlrange: list[int, int] = None,
        name: str = None,
    ) -> mjcf.Element:
        if joint.name is None:
            raise ValueError("Joint must have a name")
        actuator_mjcf = mojo.root_element.mjcf.actuator
        ctrlrange = np.array([0, 0]) if ctrlrange is None else ctrlrange
        new_actuator = actuator_mjcf.add(
            tag, joint=joint.name, ctrlrange=ctrlrange, name=name
        )
        mojo.mark_dirty()
        return new_actuator

    @staticmethod
    def create(
        mojo: Mojo,
        joint: Joint,
        ctrlrange: list[int, int] = None,
        name: str = None,
    ) -> Self:
        new_general_actuator = GeneralActuator._create(
            "general", mojo, joint, ctrlrange, name
        )
        return GeneralActuator(mojo, new_general_actuator)


class MotorActuator(GeneralActuator):
    @staticmethod
    def create(
        mojo: Mojo,
        joint: Joint,
        ctrlrange: list[int, int] = None,
        name: str = None,
    ) -> Self:
        new_motor_actuator = GeneralActuator._create(
            "motor", mojo, joint, ctrlrange, name
        )
        return MotorActuator(mojo, new_motor_actuator)


class PositionActuator(GeneralActuator):
    @staticmethod
    def create(
        mojo: Mojo,
        joint: Joint,
        ctrlrange: list[int, int] = None,
        name: str = None,
        kp: float = 1,
        kv: float = 0,
        dampratio: float = 0,
        timeconst: float = 0,
        inheritrange: float = 0,
    ) -> Self:
        new_pos_actuator = GeneralActuator._create(
            "position", mojo, joint, ctrlrange, name
        )
        new_pos_actuator.kp = kp
        new_pos_actuator.kv = kv
        new_pos_actuator.dampratio = dampratio
        new_pos_actuator.timeconst = timeconst
        new_pos_actuator.inheritrange = inheritrange
        return MotorActuator(mojo, new_pos_actuator)


class VelocityActuator(GeneralActuator):
    @staticmethod
    def create(
        mojo: Mojo,
        joint: Joint,
        ctrlrange: list[int, int] = None,
        name: str = None,
        kv: float = 1,
    ) -> Self:
        new_vel_actuator = GeneralActuator._create(
            "velocity", mojo, joint, ctrlrange, name
        )
        new_vel_actuator.kv = kv
        return MotorActuator(mojo, new_vel_actuator)
