from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mojo import Mojo
from mojo.elements import Camera


@pytest.fixture()
def mojo() -> Mojo:
    return Mojo(str(Path(__file__).parents[1] / "world.xml"))


@pytest.fixture()
def camera(mojo) -> Camera:
    return Camera.create(mojo)


def test_get_set_position(mojo: Mojo, camera: Camera):
    expected = np.array([2, 2, 2])
    camera.set_position(expected)
    assert_array_equal(camera.get_position(), expected)


def test_get_set_quaternion(mojo: Mojo, camera: Camera):
    expected = np.array([0, 1, 0, 0])
    camera.set_quaternion(expected)
    assert_array_equal(camera.get_quaternion(), expected)


def test_get_set_fovy(mojo: Mojo, camera: Camera):
    expected_fovy = 50.0
    camera.set_fovy(expected_fovy)
    mojo.step()  # This will recompile physics to check xml is correctly parsed
    assert camera.get_fovy() == expected_fovy


def test_get_set_focal(mojo: Mojo, camera: Camera):
    expected_focal = np.array([0, 0], dtype=np.float64)
    camera.set_focal(expected_focal)
    mojo.step()  # This will recompile physics to check xml is correctly parsed
    assert_array_equal(camera.get_focal(), expected_focal)


def test_get_set_sensor_size(mojo: Mojo, camera: Camera):
    expected_sensor_size = np.array([0, 0], dtype=np.float64)
    camera.set_sensor_size(expected_sensor_size)
    mojo.step()  # This will recompile physics to check xml is correctly parsed
    assert_array_equal(camera.get_sensor_size(), expected_sensor_size)


def test_get_set_focal_pixel(mojo: Mojo, camera: Camera):
    expected_focal_pixel = np.array([400, 400], dtype=np.float64)
    camera.set_focal_pixel(expected_focal_pixel)
    mojo.step()  # This will recompile physics to check xml is correctly parsed
    assert_array_equal(camera.get_focal_pixel(), expected_focal_pixel)
