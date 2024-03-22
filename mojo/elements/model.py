from __future__ import annotations

import numpy as np
from mujoco_utils import mjcf_utils

from mojo.elements.body import Body
from mojo.elements.element import MujocoElement


class MujocoModel(MujocoElement):
    @property
    def bodies(self) -> list[Body]:
        # Loop through all children
        return [
            Body(self._mojo, mjcf)
            for mjcf in mjcf_utils.safe_find_all(self.mjcf, "body")
        ]

    def set_position(self, position: np.ndarray):
        position = np.array(position)  # ensure is numpy array
        positions = np.array([b.get_position() for b in self.bodies])
        centroid_position = positions.mean(0, keepdims=True)
        positions_relative_to_centroid = positions - centroid_position
        # Now apply position to all bodies and offset
        for body, rel_pos in zip(self.bodies, positions_relative_to_centroid):
            body.set_position(rel_pos + position)

    def get_position(self) -> np.ndarray:
        positions = np.array([b.get_position() for b in self.bodies])
        return positions.mean(0)

    def set_quaternion(self, quaternion: np.ndarray):
        # NOTE: This is not mathematically correct
        # TODO: use https://github.com/christophhagen/averaging-quaternions
        quaternion = np.array(quaternion)  # ensure is numpy array
        quaternions = np.array([b.get_quaternion() for b in self.bodies])
        centroid_quaternion = quaternions.mean(0, keepdims=True)
        quaternion_relative_to_centroid = quaternions - centroid_quaternion
        # Now apply position to all bodies and offset
        for body, rel_quat in zip(self.bodies, quaternion_relative_to_centroid):
            q = rel_quat + quaternion
            body.set_quaternion(q / np.linalg.norm(q))

    def get_quaternion(self) -> np.ndarray:
        # NOTE: This is not mathematically correct
        quaternions = np.array([b.get_quaternion() for b in self.bodies])
        return quaternions.mean(0)

    def set_color(self, color: np.ndarray):
        for b in self.bodies:
            b.set_color(color)
