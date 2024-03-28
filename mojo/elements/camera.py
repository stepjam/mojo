from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mujoco_utils import mjcf_utils
from typing_extensions import Self

from mojo.elements import Body
from mojo.elements.element import MujocoElement

if TYPE_CHECKING:
    from mojo import Mojo


class Camera(MujocoElement):
    @staticmethod
    def get(
        mojo: Mojo,
        name: str,
        parent: MujocoElement = None,
    ) -> Self:
        root_mjcf = mojo.root_element.mjcf if parent is None else parent.mjcf
        mjcf = mjcf_utils.safe_find(root_mjcf, "camera", name)
        return Camera(mojo, mjcf)

    @staticmethod
    def create(
        mojo: Mojo,
        parent: MujocoElement = None,
        position: np.ndarray = None,
        quaternion: np.ndarray = None,
        fovy: float = None,
        focal: np.ndarray = None,
        sensor_size: np.ndarray = None,
    ) -> Self:
        position = np.array([0, 0, 0]) if position is None else position
        quaternion = np.array([1, 0, 0, 0]) if quaternion is None else quaternion
        if parent is not None and not isinstance(parent, Body):
            raise ValueError("Parent must be of type body for camera.")
        parent_mjcf = (
            mojo.root_element.mjcf.worldbody if parent is None else parent.mjcf
        )
        camera_params = {}
        if fovy:
            camera_params["fovy"] = fovy
        if focal:
            camera_params["focal"] = focal
        if sensor_size:
            camera_params["sensor_size"] = sensor_size
        new_camera = parent_mjcf.add(
            "camera", pos=position, quat=quaternion, **camera_params
        )
        mojo.mark_dirty()
        return Camera(mojo, new_camera)

    def set_position(self, position: np.ndarray):
        self._mojo.physics.bind(self.mjcf).pos = position
        self.mjcf.pos = position

    def get_position(self) -> np.ndarray:
        return self._mojo.physics.bind(self.mjcf).pos

    def set_focal(self, focal: np.ndarray):
        if self.mjcf.sensorsize is None:
            self.mjcf.sensorsize = np.array([0, 0])
        if self.mjcf.resolution is None:
            self.mjcf.resolution = np.array([1, 1])
        self.mjcf.focal = focal
        self._mojo.mark_dirty()

    def get_focal(self) -> np.ndarray:
        return self.mjcf.focal

    def set_sensor_size(self, sensor_size: np.ndarray):
        if self.mjcf.focal is None:
            self.mjcf.focal = np.array([0, 0])
        if self.mjcf.resolution is None:
            self.mjcf.resolution = np.array([1, 1])
        self.mjcf.sensorsize = sensor_size
        self._mojo.mark_dirty()

    def get_sensor_size(self) -> np.ndarray:
        return self.mjcf.sensorsize

    def set_fovy(self, fovy: float):
        self.mjcf.fovy = fovy
        self._mojo.mark_dirty()

    def get_fovy(self) -> np.ndarray:
        return self.mjcf.fovy
