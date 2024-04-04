import shutil
import tempfile
from pathlib import Path

import pytest

from mojo import Mojo
from mojo.elements import Geom

TEXTURE_STORE_CAPACITY = 10
MESH_STORE_CAPACITY = 10


@pytest.fixture()
def mojo() -> Mojo:
    return Mojo(
        str(Path(__file__).parents[1] / "world.xml"),
        texture_store_capacity=TEXTURE_STORE_CAPACITY,
        mesh_store_capacity=MESH_STORE_CAPACITY,
    )


@pytest.fixture()
def geom(mojo: Mojo) -> Geom:
    return Geom.create(mojo)


def test_texture_store(mojo: Mojo, geom: Geom):
    initial_count = len(mojo.root_element.mjcf.asset.texture)
    texture_path = Path(__file__).parents[1] / "assets" / "textures" / "texture00.png"
    geom.set_texture(str(texture_path))
    with pytest.warns(UserWarning):
        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(TEXTURE_STORE_CAPACITY * 2):
                temp_path = Path(temp_dir) / f"{i}{texture_path.suffix}"
                shutil.copy2(texture_path, temp_path)
                geom.set_texture(str(temp_path))
                assert (
                    len(mojo.root_element.mjcf.asset.texture) - initial_count
                    <= TEXTURE_STORE_CAPACITY
                )


def test_mesh_store(mojo: Mojo, geom: Geom):
    initial_count = len(mojo.root_element.mjcf.asset.mesh)
    mesh_path = Path(__file__).parents[1] / "assets" / "models" / "mug.obj"
    geom.set_mesh(str(mesh_path))
    with pytest.warns(UserWarning):
        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(MESH_STORE_CAPACITY * 2):
                temp_path = Path(temp_dir) / f"{i}{mesh_path.suffix}"
                shutil.copy2(mesh_path, temp_path)
                geom.set_mesh(str(temp_path))
                assert (
                    len(mojo.root_element.mjcf.asset.mesh) - initial_count
                    <= MESH_STORE_CAPACITY
                )
