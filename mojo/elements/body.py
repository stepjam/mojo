from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import quaternion
from mujoco_utils import mjcf_utils
from typing_extensions import Self

from mojo.elements import geom, joint
from mojo.elements.element import MujocoElement
from mojo.elements.utils import has_collision

if TYPE_CHECKING:
    from mojo import Mojo


class Body(MujocoElement):
    @staticmethod
    def get(
        mojo: Mojo,
        name: str,
        parent: MujocoElement = None,
    ) -> Self:
        root_mjcf = mojo.root_element.mjcf if parent is None else parent.mjcf
        mjcf = mjcf_utils.safe_find(root_mjcf, "body", name)
        return Body(mojo, mjcf)

    @staticmethod
    def create(
        mojo: Mojo,
        parent: MujocoElement = None,
        position: np.ndarray = None,
        quaternion: np.ndarray = None,
    ) -> Self:
        position = np.array([0, 0, 0]) if position is None else position
        quaternion = np.array([1, 0, 0, 0]) if quaternion is None else quaternion
        parent_mjcf = (
            mojo.root_element.mjcf.worldbody if parent is None else parent.mjcf
        )
        new_geom = parent_mjcf.add(
            "body",
            pos=position,
            quat=quaternion,
        )
        mojo.mark_dirty()
        return Body(mojo, new_geom)

    @property
    def geoms(self) -> list[geom.Geom]:
        # Loop through all children
        geoms = self.mjcf.find_all("geom") or []
        return [geom.Geom(self._mojo, mjcf) for mjcf in geoms]

    @property
    def joints(self) -> list[joint.Joint]:
        # Loop through all children
        joints = self.mjcf.find_all("joint") or []
        return [joint.Joint(self._mojo, mjcf) for mjcf in joints]

    def set_position(self, position: np.ndarray):
        position = np.array(position)  # ensure is numpy array
        if self.mjcf.freejoint is not None:
            self._mojo.physics.bind(self.mjcf.freejoint).qpos[:3] = position
        self._mojo.physics.bind(self.mjcf).pos = position
        self.mjcf.pos = position

    def get_position(self) -> np.ndarray:
        if self.mjcf.freejoint is not None:
            return self._mojo.physics.bind(self.mjcf.freejoint).qpos[:3].copy()
        return self._mojo.physics.bind(self.mjcf).pos.copy()

    def set_quaternion(self, quaternion: np.ndarray):
        # wxyz
        quaternion = np.array(quaternion)  # ensure is numpy array
        if self.mjcf.freejoint is not None:
            self._mojo.physics.bind(self.mjcf.freejoint).qpos[3:] = quaternion
        self._mojo.physics.bind(self.mjcf).quat = quaternion
        self.mjcf.quat = quaternion

    def get_quaternion(self) -> np.ndarray:
        if self.mjcf.freejoint is not None:
            return self._mojo.physics.bind(self.mjcf.freejoint).qpos[3:].copy()
        return self._mojo.physics.bind(self.mjcf).quat.copy()

    def set_euler(self, euler: np.ndarray):
        self.set_quaternion(
            quaternion.as_float_array(
                quaternion.from_euler_angles(euler[0], euler[1], euler[2])
            )
        )

    def set_color(self, color: np.ndarray):
        for b in self.geoms:
            b.set_color(color)

    def set_texture(self, texture_path: str):
        for b in self.geoms:
            b.set_texture(texture_path)

    def set_kinematic(self, value: bool):
        if value and not self.is_kinematic():
            self.mjcf.add("freejoint")
            self._mojo.mark_dirty()

    def is_kinematic(self) -> bool:
        return self.mjcf.freejoint is not None

    def set_collidable(self, value: bool):
        for g in self.geoms:
            g.set_collidable(value)

    def is_collidable(self) -> bool:
        return len(self.geoms) > 0 and self.geoms[0].is_collidable()

    def has_collided(self, other: Body = None):
        if other is not None and (not other.is_kinematic() and not self.is_kinematic()):
            warnings.warn("You are checking collisions of two non-kinematic bodies.")
        # If None, return true if there is any contact
        if other is None:
            return len(self._mojo.physics.data.contact) > 0
        this_object_id = self._mojo.physics.bind(self.mjcf).element_id
        other_object_id = self._mojo.physics.bind(other.mjcf).element_id
        return has_collision(self._mojo.physics, other_object_id, this_object_id)
