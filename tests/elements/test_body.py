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
def body(mojo: Mojo) -> Body:
    return Body.create(mojo)


def test_get_geoms(mojo: Mojo, body: Body):
    new_geom = Geom.create(mojo, parent=body)
    assert len(body.geoms) == 1
    assert new_geom == body.geoms[0]


def test_get_set_position(mojo: Mojo, body: Body):
    expected = np.array([2, 2, 2])
    body.set_position(expected)
    assert_array_equal(body.get_position(), expected)


def test_get_set_quaternion(mojo: Mojo, body: Body):
    expected = np.array([0, 1, 0, 0])
    body.set_quaternion(expected)
    assert_array_equal(body.get_quaternion(), expected)
