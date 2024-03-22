from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mujoco_utils import mjcf_utils
from typing_extensions import Self

from mojo.elements import geom
from mojo.elements.element import MujocoElement

if TYPE_CHECKING:
    from mojo import Mojo


class Body(MujocoElement):
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
        return [
            geom.Geom(self._mojo, mjcf)
            for mjcf in mjcf_utils.safe_find_all(self.mjcf, "geom")
        ]

    def set_position(self, position: np.ndarray):
        position = np.array(position)  # ensure is numpy array
        self._mojo.physics.bind(self.mjcf).pos = position
        self.mjcf.pos = position

    def get_position(self) -> np.ndarray:
        return self._mojo.physics.bind(self.mjcf).pos

    def set_quaternion(self, quaternion: np.ndarray):
        # wxyz
        quaternion = np.array(quaternion)  # ensure is numpy array
        self._mojo.physics.bind(self.mjcf).quat = quaternion
        self.mjcf.quat = quaternion

    def get_quaternion(self) -> np.ndarray:
        return self._mojo.physics.bind(self.mjcf).quat

    def set_color(self, color: np.ndarray):
        for b in self.geoms:
            b.set_color(color)

    def set_texture(self, texture_path: str):
        for b in self.geoms:
            b.set_texture(texture_path)
