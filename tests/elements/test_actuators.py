from pathlib import Path

import pytest

from mojo import Mojo
from mojo.elements.actuators import (
    GeneralActuator,
    MotorActuator,
    PositionActuator,
    VelocityActuator,
)
from mojo.elements.joint import Joint


@pytest.fixture()
def mojo() -> Mojo:
    return Mojo(str(Path(__file__).parents[1] / "world.xml"))


@pytest.fixture()
def general_actuator(mojo: Mojo) -> GeneralActuator:
    j = Joint.create(mojo, name="my_joint")
    return GeneralActuator.create(mojo, joint=j)


@pytest.fixture()
def motor_actuator(mojo: Mojo) -> MotorActuator:
    j = Joint.create(mojo, name="my_joint")
    return MotorActuator.create(mojo, joint=j)


@pytest.fixture()
def position_actuator(mojo: Mojo) -> PositionActuator:
    j = Joint.create(mojo, name="my_joint")
    return PositionActuator.create(mojo, joint=j)


@pytest.fixture()
def velocity_actuator(mojo: Mojo) -> VelocityActuator:
    j = Joint.create(mojo, name="my_joint")
    return VelocityActuator.create(mojo, joint=j)


def test_get_general_actuator(mojo: Mojo, general_actuator: GeneralActuator):
    acts = mojo.actuators
    assert len(acts) == 1
    assert acts[0] == general_actuator


def test_get_motor_actuator(mojo: Mojo, motor_actuator: MotorActuator):
    acts = mojo.actuators
    assert len(acts) == 1
    assert acts[0] == motor_actuator


def test_get_position_actuator(mojo: Mojo, position_actuator: PositionActuator):
    acts = mojo.actuators
    assert len(acts) == 1
    assert acts[0] == position_actuator


def test_get_velocity_actuator(mojo: Mojo, velocity_actuator: VelocityActuator):
    acts = mojo.actuators
    assert len(acts) == 1
    assert acts[0] == velocity_actuator
