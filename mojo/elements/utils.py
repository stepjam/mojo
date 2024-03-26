import uuid

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
    uid = str(uuid.uuid4())
    texture = mjcf_model.asset.add(
        "texture", name=f"texture_{uid}", file=path, type=mapping.value
    )
    material = mjcf_model.asset.add(
        "material",
        name=f"material_{uid}",
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
