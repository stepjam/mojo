from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mojo import Mojo
from mojo.elements import Body, Geom


@pytest.fixture()
def mojo() -> Mojo:
    return Mojo(str(Path(__file__).parents[1] / "world.xml"))


@pytest.fixture()
def geom(mojo: Mojo) -> Geom:
    return Geom.create(mojo)


def test_get_set_position(mojo: Mojo, geom: Geom):
    expected = np.array([2, 2, 2])
    geom.set_position(expected)
    assert_array_equal(geom.get_position(), expected)


def test_get_set_color(mojo: Mojo, geom: Geom):
    expected = np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32)
    geom.set_color(expected)
    assert_array_equal(geom.get_color(), expected)


def test_set_texture(mojo: Mojo, geom: Geom):
    # just test that there are no exceptions
    geom.set_texture(
        str(Path(__file__).parents[1] / "assets" / "textures" / "texture00.png")
    )


def test_get_parent(mojo: Mojo, geom: Geom):
    assert geom.parent is not None


def test_get_set_collidable(mojo: Mojo, geom: Geom):
    geom.set_collidable(True)
    assert geom.is_collidable()
    geom.set_collidable(False)
    assert not geom.is_collidable()


def test_has_collided(mojo: Mojo):
    body0 = Body.create(mojo)
    body1 = Body.create(mojo)
    body2 = Body.create(mojo, position=np.array([1, 1, 1]))
    body1.set_kinematic(True)
    body2.set_kinematic(True)
    geom0 = Geom.create(mojo, body0)
    geom1 = Geom.create(mojo, body1)
    geom2 = Geom.create(mojo, body2)
    assert not geom0.has_collided(geom2)
    assert geom0.has_collided(geom1)
