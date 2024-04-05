from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
from mujoco_utils import mjcf_utils
from typing_extensions import Self

from mojo.elements import body
from mojo.elements.consts import SiteType, TextureMapping
from mojo.elements.element import MujocoElement
from mojo.elements.utils import load_texture

if TYPE_CHECKING:
    from mojo import Mojo
    from mojo.elements.body import Body


class Site(MujocoElement):
    @staticmethod
    def get(
        mojo: Mojo,
        name: str,
        parent: MujocoElement = None,
    ) -> Self:
        root_mjcf = mojo.root_element.mjcf if parent is None else parent.mjcf
        mjcf = mjcf_utils.safe_find(root_mjcf, "site", name)
        return Site(mojo, mjcf)

    @staticmethod
    def create(
        mojo: Mojo,
        parent: body.Body = None,
        size: np.ndarray = None,
        position: np.ndarray = None,
        quaternion: np.ndarray = None,
        color: np.ndarray = None,
        site_type: SiteType = SiteType.SPHERE,
        group: int = 1,
    ) -> Self:
        position = np.array([0, 0, 0]) if position is None else position
        quaternion = np.array([1, 0, 0, 0]) if quaternion is None else quaternion
        size = np.array([0.1, 0.1, 0.1]) if size is None else size
        color = np.array([1, 1, 1, 1]) if color is None else color
        parent = body.Body.create(mojo) if parent is None else parent
        new_geom = parent.mjcf.add(
            "site",
            type=site_type.value,
            pos=position,
            quat=quaternion,
            size=size,
            rgba=color,
            group=group,
        )
        new_site_obj = Site(mojo, new_geom)
        mojo.mark_dirty()
        return new_site_obj

    @property
    def parent(self) -> "Body":
        # Have to do this due to circular import
        from mojo.elements.body import Body

        return Body(self._mojo, self.mjcf.parent)

    def set_position(self, position: np.ndarray):
        position = np.array(position)  # ensure is numpy array
        if self.mjcf.parent.freejoint:
            self._mojo.physics.bind(self.mjcf.parent.freejoint).qpos[:3] = position
        self._mojo.physics.bind(self.mjcf).pos = position
        self.mjcf.pos = position

    def get_position(self) -> np.ndarray:
        if self.mjcf.parent.freejoint:
            return self._mojo.physics.bind(self.mjcf.parent.freejoint).qpos[:3].copy()
        return self._mojo.physics.bind(self.mjcf).xpos

    def set_quaternion(self, quaternion: np.ndarray):
        # wxyz
        quaternion = np.array(quaternion)  # ensure is numpy array
        if self.mjcf.parent.freejoint is not None:
            self._mojo.physics.bind(self.mjcf.parent.freejoint).qpos[3:] = quaternion
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quaternion)
        self._mojo.physics.bind(self.mjcf).xmat = mat
        self.mjcf.quat = quaternion

    def get_quaternion(self) -> np.ndarray:
        if self.mjcf.parent.freejoint is not None:
            return self._mojo.physics.bind(self.mjcf.parent.freejoint).qpos[3:].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self._mojo.physics.bind(self.mjcf).xmat)
        return quat

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
        key_name = f"{texture_path}_{mapping.value}"
        material = self._mojo.get_material(key_name)
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
            self._mojo.store_material(key_name, material)
        self.mjcf.material = material
        if self.mjcf.rgba is None:
            # Have a default white color for texture
            self.set_color(np.ones(4))
        self._mojo.mark_dirty()
