from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mujoco_utils import mjcf_utils
from typing_extensions import Self

from mojo.elements import body
from mojo.elements.consts import GeomType, TextureMapping
from mojo.elements.element import MujocoElement
from mojo.elements.utils import has_collision, load_texture

if TYPE_CHECKING:
    from mojo import Mojo
    from mojo.elements.body import Body


class Geom(MujocoElement):
    @staticmethod
    def get(
        mojo: Mojo,
        name: str,
        parent: MujocoElement = None,
    ) -> Self:
        root_mjcf = mojo.root_element.mjcf if parent is None else parent.mjcf
        mjcf = mjcf_utils.safe_find(root_mjcf, "geom", name)
        return Geom(mojo, mjcf)

    @staticmethod
    def create(
        mojo: Mojo,
        parent: body.Body = None,
        size: np.ndarray = None,
        position: np.ndarray = None,
        quaternion: np.ndarray = None,
        color: np.ndarray = None,
        geom_type: GeomType = GeomType.BOX,
    ) -> Self:
        position = np.array([0, 0, 0]) if position is None else position
        quaternion = np.array([1, 0, 0, 0]) if quaternion is None else quaternion
        size = np.array([0.1, 0.1, 0.1]) if size is None else size
        color = np.array([1, 1, 1, 1]) if color is None else color
        parent = body.Body.create(mojo) if parent is None else parent
        new_geom = parent.mjcf.add(
            "geom",
            type=geom_type.value,
            pos=position,
            quat=quaternion,
            size=size,
            rgba=color,
        )
        mojo.mark_dirty()
        return Geom(mojo, new_geom)

    @property
    def parent(self) -> "Body":
        # Have to do this due to circular import
        from mojo.elements.body import Body

        return Body(self._mojo, self.mjcf.parent)

    def set_position(self, position: np.ndarray):
        position = np.array(position)  # ensure is numpy array
        self._mojo.physics.bind(self.mjcf).pos = position
        self.mjcf.pos = position

    def get_position(self) -> np.ndarray:
        return self._mojo.physics.bind(self.mjcf).pos

    def set_color(self, color: np.ndarray):
        color = np.array(color)
        if len(color) == 3:
            color = np.concatenate([color, [1]])  # add alpha
        self._mojo.physics.bind(self.mjcf).rgba = color
        self.mjcf.rgba = color

    def get_color(self) -> np.ndarray:
        return np.array(self._mojo.physics.bind(self.mjcf).rgba)

    def set_texture(
        self,
        texture_path: str,
        mapping: TextureMapping = TextureMapping.CUBE,
        tex_repeat: np.ndarray = None,
        tex_uniform: bool = False,
        emission: float = 0.0,
        specular: float = 0.0,
        shininess: float = 0.0,
        reflectance: float = 0.0,
        color: np.ndarray = None,
    ):
        # First check if we have loaded this texture
        material = self._mojo.get_material(texture_path)
        if material is None:
            material = load_texture(
                self._mojo.root_element.mjcf,
                texture_path,
                mapping,
                tex_repeat,
                tex_uniform,
                emission,
                specular,
                shininess,
                reflectance,
                color,
            )
            self._mojo.store_material(texture_path, material)
        self.mjcf.material = material
        if self.mjcf.rgba is None:
            # Have a default white color for texture
            self.set_color(np.ones(4))
        self._mojo.mark_dirty()

    def set_collidable(self, value: bool):
        self.mjcf.contype = int(value)
        self.mjcf.conaffinity = int(value)
        self._mojo.physics.bind(self.mjcf).contype = int(value)
        self._mojo.physics.bind(self.mjcf).conaffinity = int(value)

    def is_collidable(self) -> bool:
        return (
            self._mojo.physics.bind(self.mjcf).contype == 1
            and self._mojo.physics.bind(self.mjcf).conaffinity == 1
        )

    def has_collided(self, other: Geom = None):
        # If None, return true if there is any contact
        if other is None:
            return len(self._mojo.physics.data.contact) > 0
        this_object_id = self._mojo.physics.bind(self.mjcf).element_id
        other_object_id = self._mojo.physics.bind(other.mjcf).element_id
        return has_collision(self._mojo.physics, other_object_id, this_object_id)
