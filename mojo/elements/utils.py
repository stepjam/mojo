import uuid

import numpy as np
from dm_control import mjcf

from mojo.elements.consts import TextureMapping


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
