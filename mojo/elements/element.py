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


def _remove_all_joints(elem: mjcf.Element):
    if hasattr(elem, "freejoint") and elem.freejoint is not None:
        elem.freejoint.remove()
    if hasattr(elem, "joint") and len(elem.joint) > 0:
        for joint in elem.joint:
            joint.remove()


def _find_freejoint(elem: mjcf.Element):
    if elem.parent is None:
        # Root of tree
        return None
    if free_joint := getattr(elem, "freejoint", None):
        return free_joint
    return _find_freejoint(elem.parent)


class MujocoElement(ABC):
    def __init__(self, mojo: Mojo, mjcf_elem: mjcf.Element):
        self._mojo = mojo
        self._mjcf_elem = mjcf_elem

    @property
    def mjcf(self):
        return self._mjcf_elem

    @property
    def id(self):
        return self._mojo.physics.bind(self.mjcf).element_id

    @property
    def name(self):
        return self.mjcf.name

    def __eq__(self, other):
        return (
            isinstance(other, MujocoElement)
            and self.mjcf.full_identifier == other.mjcf.full_identifier
        )

    def __str__(self):
        return self._mjcf_elem.to_xml_string()


class TransformElement(MujocoElement):
    def set_position(self, position: np.ndarray, reset_dynamics: bool = True):
        position = np.array(position)  # ensure is numpy array
        if freejoint := _find_freejoint(self.mjcf):
            freejoint = self._mojo.physics.bind(freejoint)
            freejoint.qpos[:3] = position
            if reset_dynamics:
                freejoint.qvel *= 0
                freejoint.qacc *= 0
        else:
            self._mojo.physics.bind(self.mjcf).pos = position
        self.mjcf.pos = position

    def get_position(self) -> np.ndarray:
        # if the element has a free joint (and thus is a body), then access qpos
        if freejoint := _find_freejoint(self.mjcf):
            return self._mojo.physics.bind(freejoint).qpos[:3].copy()
        return self._mojo.physics.bind(self.mjcf).xpos.copy()

    def set_quaternion(self, quaternion: np.ndarray, reset_dynamics: bool = True):
        # wxyz
        quaternion = np.array(quaternion)  # ensure is numpy array
        if freejoint := _find_freejoint(self.mjcf):
            freejoint = self._mojo.physics.bind(freejoint)
            freejoint.qpos[3:] = quaternion
            if reset_dynamics:
                freejoint.qvel *= 0
                freejoint.qacc *= 0
        binded = self._mojo.physics.bind(self.mjcf)
        if binded.quat is not None:
            binded.quat = quaternion
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quaternion)
        binded.xmat = mat
        self.mjcf.quat = quaternion

    def get_quaternion(self) -> np.ndarray:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self._mojo.physics.bind(self.mjcf).xmat)
        return quat

    def is_kinematic(self) -> bool:
        return _is_kinematic(self.mjcf)

    def remove_all_joints(self):
        _remove_all_joints(self.mjcf)
