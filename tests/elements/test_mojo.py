from pathlib import Path

import pytest

from mojo import Mojo
from mojo.elements import Body


@pytest.fixture()
def mojo() -> Mojo:
    return Mojo(str(Path(__file__).parents[1] / "world.xml"))


def load_model(mojo: Mojo, model_name: str, handle_freejoints: bool) -> Body:
    body = mojo.load_model(
        str(Path(__file__).parents[1] / "assets" / "models" / model_name),
        handle_freejoints=handle_freejoints,
    )
    _ = mojo.physics
    return body


def test_load_freejoint(mojo: Mojo):
    load_model(mojo, "sphere.xml", True)


def test_load_freejoint_raises_by_default(mojo: Mojo):
    with pytest.raises(ValueError):
        load_model(mojo, "sphere.xml", False)


def test_load_freejoint_hierarchy(mojo: Mojo):
    sphere_and_box = load_model(mojo, "sphere_and_box.xml", True)
    for joint in sphere_and_box.joints:
        assert joint.mjcf.tag == "freejoint"
        assert joint.mjcf.parent.parent.tag == "worldbody"
