from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mojo import Mojo
from mojo.elements.consts import LightType
from mojo.elements.light import Light


@pytest.fixture()
def mojo() -> Mojo:
    return Mojo(str(Path(__file__).parents[1] / "world.xml"))


@pytest.fixture()
def light(mojo) -> Light:
    return Light.create(mojo)


def test_get_set_position(mojo, light):
    expected = np.array([2, 2, 2])
    light.set_position(expected)
    assert_array_equal(light.get_position(), expected)


def test_get_set_active(mojo, light):
    expected = False
    light.set_active(expected)
    assert light.is_active() == expected
    expected = True
    light.set_active(expected)
    assert light.is_active() == expected


def test_get_set_ambient(mojo, light):
    expected = np.array([0.8, 0.8, 0.8])
    light.set_ambient(expected)
    assert_array_equal(light.get_ambient(), expected)


def test_get_set_diffuse(mojo, light):
    expected = np.array([0.8, 0.8, 0.8])
    light.set_diffuse(expected)
    assert_array_equal(light.get_diffuse(), expected)


def test_get_set_specular(mojo, light):
    expected = np.array([0.8, 0.8, 0.8])
    light.set_specular(expected)
    assert_array_equal(light.get_specular(), expected)


def test_get_set_direction(mojo, light):
    expected = np.array([0.8, 0.8, 0.8])
    light.set_direction(expected)
    assert_array_equal(light.get_direction(), expected)


def test_get_light_type(mojo, light):
    assert light.get_light_type(), LightType.SPOTLIGHT
