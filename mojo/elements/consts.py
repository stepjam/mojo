from enum import Enum


class GeomType(Enum):
    PLANE = "plane"
    HFIELD = "hfield"
    SPHERE = "sphere"
    CAPSULE = "capsule"
    ELLIPSOID = "ellipsoid"
    CYLINDER = "cylinder"
    BOX = "box"
    MESH = "mesh"
    SDF = "sdf"


class TextureMapping(Enum):
    PLANAR = "2d"
    CUBE = "cube"
    SKYBOX = "skybox"
