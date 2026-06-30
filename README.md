# mojo-mujoco-wrapper

`mojo-mujoco-wrapper` is a Python library that wraps [MuJoCo](https://mujoco.org/) with a simple object-oriented API. You create and control simulation elements (bodies, geoms, joints, cameras, lights) as Python objects instead of working with MuJoCo's low-level arrays directly. The library handles model construction, recompilation, and physics stepping so you can focus on building scenes and controllers.

## Installation

```bash
pip install mojo-mujoco-wrapper
```

## Usage

### Basic scene

```python
from mojo import Mojo
from mojo.elements import Body, Geom, Joint
from mojo.elements.consts import JointType

mojo = Mojo("world.xml")

# Create a body with a box geom and a hinge joint
body = Body.create(mojo, position=[0, 0, 1])
geom = Geom.create(mojo, parent=body)
joint = Joint.create(mojo, parent=body, joint_type=JointType.HINGE)

# Step the simulation
for _ in range(100):
    mojo.step()

print(body.get_position())
```

### Camera

```python
from mojo.elements import Camera
import numpy as np

camera = Camera.create(mojo, position=[0, -2, 1])
camera.set_fovy(60.0)
camera.set_quaternion(np.array([1, 0, 0, 0]))
```

### Kinematic bodies

```python
body = Body.create(mojo, position=[0, 0, 0.5])
Geom.create(mojo, parent=body)
body.set_kinematic(True)  # body moves with set_position, ignores physics forces

body.set_position([1, 0, 0.5])
mojo.step()
```

### Getting elements by name

```python
body = Body.get(mojo, name="my_body")
camera = Camera.get(mojo, name="top_cam")
```

## Development

### Setup

```bash
git clone https://github.com/stepjam/mojo
cd mojo
pip install -e ".[dev]"
```

### Running tests

```bash
pytest tests/
```

### Contributing

1. Fork the repo and create a branch off `main`.
2. Add tests for any new elements or behavior.
3. Open a pull request against `main`.
