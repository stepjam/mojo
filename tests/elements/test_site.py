from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mojo import Mojo
from mojo.elements import Site


@pytest.fixture()
def mojo() -> Mojo:
    return Mojo(str(Path(__file__).parents[1] / "world.xml"))


@pytest.fixture()
def site(mojo: Mojo) -> Site:
    return Site.create(mojo)


def test_get_set_position(mojo: Mojo, site: Site):
    expected = np.array([2, 2, 2])
    site.set_position(expected)
    assert_array_equal(site.get_position(), expected)


def test_get_set_quaternion(mojo: Mojo, site: Site):
    expected = np.array([0, 1, 0, 0])
    site.set_quaternion(expected)
    assert_array_equal(site.get_quaternion(), expected)


def test_get_set_matrix(mojo: Mojo, site: Site):
    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    site.set_matrix(expected)
    assert_array_equal(site.get_matrix(), expected)


def test_get_set_color(mojo: Mojo, site: Site):
    expected = np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32)
    site.set_color(expected)
    assert_array_equal(site.get_color(), expected)


def test_set_texture(mojo: Mojo, site: Site):
    # just test that there are no exceptions
    site.set_texture(
        str(Path(__file__).parents[1] / "assets" / "textures" / "texture00.png")
    )


def test_get_parent(mojo: Mojo, site: Site):
    assert site.parent is not None
