import codecs
import os

import setuptools


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


core_requirements = [
    "mujoco>=2.3.3",
    "numpy>=1.26",
    "dm_control>=1.0.0",
    "mujoco_utils>=0.0.6",
    "numpy-quaternion>=2024.0.13",
]

setuptools.setup(
    version=get_version("mojo/__init__.py"),
    name="mojo-mujoco-wrapper",
    author="Stephen James",
    author_email="stepjamuk@gmail.com",
    description=(
        "Python wrapper for building and controlling MuJoCo physics simulations."
    ),
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/stepjam/mojo",
    license="MIT",
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    install_requires=core_requirements,
    extras_require={
        "dev": ["pre-commit", "pytest"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
