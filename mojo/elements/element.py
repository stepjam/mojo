from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from dm_control import mjcf

if TYPE_CHECKING:
    from mojo import Mojo


def _is_kinematic(elem: mjcf.Element):
    if elem.parent is None:
        # Root of tree
        return False
    has_freejoint = hasattr(elem, "freejoint") and elem.freejoint is not None
    has_joints = hasattr(elem, "joint") and len(elem.joint) > 0
    return has_freejoint or has_joints or _is_kinematic(elem.parent)


class MujocoElement(ABC):
    def __init__(self, mojo: Mojo, mjcf_elem: mjcf.RootElement):
        self._mojo = mojo
        self._mjcf_elem = mjcf_elem

    @property
    def mjcf(self):
        return self._mjcf_elem

    @property
    def id(self):
        return self._mojo.physics.bind(self.mjcf).element_id

    def set_position(self, position: np.ndarray):
        position = np.array(position)  # ensure is numpy array
        if hasattr(self.mjcf, "freejoint") and self.mjcf.freejoint is not None:
            self._mojo.physics.bind(self.mjcf.freejoint).qpos[:3] = position
        else:
            self._mojo.physics.bind(self.mjcf).pos = position
        self.mjcf.pos = position

    def get_position(self) -> np.ndarray:
        # if the element has a free joint (and thus is a body), then access qpos
        if hasattr(self.mjcf, "freejoint") and self.mjcf.freejoint is not None:
            return self._mojo.physics.bind(self.mjcf.freejoint).qpos[:3].copy()
        return self._mojo.physics.bind(self.mjcf).xpos.copy()

    def set_quaternion(self, quaternion: np.ndarray):
        # wxyz
        quaternion = np.array(quaternion)  # ensure is numpy array
        if hasattr(self.mjcf, "freejoint") and self.mjcf.freejoint is not None:
            self._mojo.physics.bind(self.mjcf.freejoint).qpos[3:] = quaternion
        binded = self._mojo.physics.bind(self.mjcf)
        if binded.quat is not None:
            binded.quat = quaternion
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quaternion)
        self._mojo.physics.bind(self.mjcf).xmat = mat
        self.mjcf.quat = quaternion

    def get_quaternion(self) -> np.ndarray:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self._mojo.physics.bind(self.mjcf).xmat)
        return quat

    def is_kinematic(self) -> bool:
        return _is_kinematic(self.mjcf)

    def __eq__(self, other):
        return (
            isinstance(other, MujocoElement)
            and self.mjcf.full_identifier == other.mjcf.full_identifier
        )
