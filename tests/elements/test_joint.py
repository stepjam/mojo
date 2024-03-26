from pathlib import Path

import pytest

from mojo import Mojo
from mojo.elements import Body, Geom
from mojo.elements.consts import JointType
from mojo.elements.joint import Joint


@pytest.fixture()
def mojo() -> Mojo:
    return Mojo(str(Path(__file__).parents[1] / "world.xml"))


@pytest.fixture()
def geom(mojo: Mojo) -> Body:
    return Geom.create(mojo)


def test_slide_joint(mojo: Mojo, geom: Geom):
    new_joint = Joint.create(mojo, parent=geom.parent, joint_type=JointType.SLIDE)
    assert new_joint.get_joint_position() == 0
    assert new_joint.get_joint_velocity() == 0
