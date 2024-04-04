import uuid
import warnings
from collections import OrderedDict
from typing import Optional

import numpy as np
from dm_control import mjcf

from mojo.elements.consts import TextureMapping

# Default minimum distance between two geoms for them to be considered in collision.
_DEFAULT_COLLISION_MARGIN: float = 1e-8


def has_collision(
    physics,
    collision_geom_id_1: int,
    collision_geom_id_2: int,
    margin: float = _DEFAULT_COLLISION_MARGIN,
) -> bool:
    """Check collision between two objects by geometry id."""
    for contact in physics.data.contact:
        if contact.dist > margin:
            continue

        if (
            contact.geom1 == collision_geom_id_1
            and contact.geom2 == collision_geom_id_2
        ) or (
            contact.geom2 == collision_geom_id_1
            and contact.geom1 == collision_geom_id_2
        ):
            return True

    return False


def load_texture(
    mjcf_model: mjcf.RootElement,
    path: str,
    mapping: TextureMapping = TextureMapping.CUBE,
    tex_repeat: np.ndarray = None,
    tex_uniform: bool = False,
    emission: float = 0.0,
    specular: float = 0.0,
    shininess: float = 0.0,
    reflectance: float = 0.0,
    color: np.ndarray = None,
) -> mjcf.Element:
    tex_repeat = np.array([1, 1]) if tex_repeat is None else tex_repeat
    color = np.array([1, 1, 1, 1]) if color is None else color
    name = f"{uuid.uuid4()}_{mapping.value}"
    texture = mjcf_model.asset.add(
        "texture", name=f"texture_{name}", file=path, type=mapping.value
    )
    material = mjcf_model.asset.add(
        "material",
        name=f"material_{name}",
        texture=texture,
        texrepeat=tex_repeat,
        texuniform=str(tex_uniform).lower(),
        emission=emission,
        specular=specular,
        shininess=shininess,
        reflectance=reflectance,
        rgba=color,
    )
    return material


def load_mesh(
    mjcf_model: mjcf.RootElement, path: str, scale: np.ndarray
) -> mjcf.Element:
    scale = np.array([1, 1, 1]) if scale is None else scale
    uid = str(uuid.uuid4())
    mesh = mjcf_model.asset.add("mesh", name=f"mesh_{uid}", file=path, scale=scale)
    return mesh


class AssetStore:
    """Container for Mujoco assets."""

    DEFAULT_CAPACITY = 32

    def __init__(self, capacity: Optional[int] = None):
        self._store: OrderedDict[str, mjcf.Element] = OrderedDict()
        self._capacity = capacity

    def get(self, path: str) -> Optional[mjcf.Element]:
        """Get MJCF asset by path."""
        return self._store.get(path, None)

    def remove(self, path: str) -> None:
        """Remove MJCF asset by path."""
        if path in self._store:
            asset = self._store.pop(path)
            self._unload_asset(asset)

    def store(self, path: str, asset_mjcf: mjcf.Element) -> None:
        """Add new MJCF asset."""
        self._store[path] = asset_mjcf
        if self._capacity and len(self._store) > self._capacity:
            warnings.warn(
                f"The capacity of the store ({self._capacity}) has been exceeded."
                f"Removing the oldest asset.",
                UserWarning,
            )
            _, asset = self._store.popitem(last=False)
            self._unload_asset(asset)

    @staticmethod
    def _unload_asset(asset: mjcf.Element) -> None:
        if asset.tag == "material":
            asset.texture.remove()
        asset.remove()
