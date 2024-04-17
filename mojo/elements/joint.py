from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mujoco_utils import mjcf_utils
from typing_extensions import Self

from mojo.elements import body
from mojo.elements.consts import JointType
from mojo.elements.element import MujocoElement

if TYPE_CHECKING:
    from mojo import Mojo


class Joint(MujocoElement):
    @staticmethod
    def get(
        mojo: Mojo,
        name: str,
        parent: MujocoElement = None,
    ) -> Self:
        root_mjcf = mojo.root_element.mjcf if parent is None else parent.mjcf
        mjcf = mjcf_utils.safe_find(root_mjcf, "joint", name)
        return Joint(mojo, mjcf)

    @staticmethod
    def create(
        mojo: Mojo,
        parent: body.Body = None,
        position: np.ndarray = None,
        axis: np.ndarray = None,
        range: np.ndarray = None,
        joint_type: JointType = JointType.HINGE,
        stiffness: float = 0,
        springref: float = 0,
    ) -> Self:
        position = np.array([0, 0, 0]) if position is None else position
        axis = np.array([1, 0, 0]) if axis is None else axis
        range = np.array([0.0, 0.0]) if range is None else range
        parent = body.Body.create(mojo) if parent is None else parent
        new_geom = parent.mjcf.add(
            "joint",
            type=joint_type.value,
            pos=position,
            axis=axis,
            range=range,
            stiffness=stiffness,
            springref=springref,
        )
        mojo.mark_dirty()
        return Joint(mojo, new_geom)

    def get_joint_position(self) -> float:
        """Get current joint position."""
        return float(self._mojo.physics.bind(self.mjcf).qpos.item())

    def set_joint_position(self, value: float):
        self._mojo.physics.bind(self.mjcf).qpos *= 0
        self._mojo.physics.bind(self.mjcf).qpos += value
        self._mojo.mark_dirty()

    def get_joint_velocity(self) -> float:
        """Get current joint velocity."""
        return float(self._mojo.physics.bind(self.mjcf).qvel.item())
