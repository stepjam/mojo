from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mujoco_utils import mjcf_utils
from typing_extensions import Self

from mojo.elements import Body
from mojo.elements.consts import LightType
from mojo.elements.element import TransformElement

if TYPE_CHECKING:
    from mojo import Mojo


class Light(TransformElement):
    @staticmethod
    def get(
        mojo: Mojo,
        name: str,
        parent: TransformElement = None,
    ) -> Self:
        root_mjcf = mojo.root_element.mjcf if parent is None else parent.mjcf
        mjcf = mjcf_utils.safe_find(root_mjcf, "light", name)
        return Light(mojo, mjcf)

    @staticmethod
    def create(
        mojo: Mojo,
        parent: TransformElement = None,
        position: np.ndarray = None,
        direction: np.ndarray = None,
        light_type: LightType = LightType.SPOTLIGHT,
        shadows: bool = True,
        ambient: np.ndarray = None,
        diffuse: np.ndarray = None,
        specular: np.ndarray = None,
    ) -> Self:
        position = np.array([0, 0, 2]) if position is None else position
        direction = np.array([0, 0, -1]) if direction is None else direction
        ambient = np.array([0.1, 0.1, 0.1]) if ambient is None else ambient
        diffuse = np.array([0.4, 0.4, 0.4]) if diffuse is None else diffuse
        specular = np.array([0.5, 0.5, 0.5]) if specular is None else specular
        if parent is not None and not isinstance(parent, Body):
            raise ValueError("Parent must be of type body for lights.")
        parent_mjcf = (
            mojo.root_element.mjcf.worldbody if parent is None else parent.mjcf
        )
        new_light = parent_mjcf.add(
            "light",
            pos=position,
            dir=direction,
            directional=light_type == LightType.DIRECTIONAL,
            castshadow=shadows,
            ambient=ambient,
            diffuse=diffuse,
            specular=specular,
        )
        mojo.mark_dirty()
        return Light(mojo, new_light)

    def set_active(self, value: bool):
        self.mjcf.active = value
        self._mojo.physics.bind(self.mjcf).active = value

    def is_active(self) -> bool:
        return self.mjcf.active == "true"

    def set_ambient(self, color: np.ndarray):
        self.mjcf.ambient = color
        self._mojo.physics.bind(self.mjcf).ambient = color

    def get_ambient(self) -> np.ndarray:
        return self.mjcf.ambient

    def set_diffuse(self, color: np.ndarray):
        self.mjcf.diffuse = color
        self._mojo.physics.bind(self.mjcf).diffuse = color

    def get_diffuse(self) -> np.ndarray:
        return self.mjcf.diffuse

    def set_specular(self, color: np.ndarray):
        self.mjcf.specular = color
        self._mojo.physics.bind(self.mjcf).specular = color

    def get_specular(self) -> np.ndarray:
        return self.mjcf.specular

    def set_direction(self, direction: np.ndarray):
        self.mjcf.dir = direction
        self._mojo.physics.bind(self.mjcf).dir = direction

    def get_direction(self) -> np.ndarray:
        return self.mjcf.dir

    def set_shadows(self, value: bool):
        self.mjcf.castshadow = value
        self._mojo.physics.bind(self.mjcf).castshadow = value

    def is_using_shadows(self) -> bool:
        return self.mjcf.castshadow == "true"

    def get_light_type(self) -> LightType:
        return LightType.DIRECTIONAL if self.mjcf.directional else LightType.SPOTLIGHT
