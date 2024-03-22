"""Example showing:

 - Spawning of objects
 - Moving objects
 - Using passive viewer
 - Setting colors
 - Add a light
"""
import time
from pathlib import Path

import numpy as np

from mojo import Mojo
from mojo.elements import Body, Geom
from mojo.elements.light import Light

mojo = Mojo(str(Path(__file__).parent / "world.xml"), 0.02)
mojo.launch_viewer(passive=True)
mojo.set_headlight(active=False)  # Turn off main light
body = Body.create(mojo)
light = Light.create(mojo, ambient=np.array([1, 0, 0]))
for i in range(2000):
    time.sleep(0.005)
    mojo.sync_passive_viewer()
    if i % 500 == 0:
        print("Adding cube!")
        cube = Geom.create(
            mojo,
            parent=body,
            position=np.random.uniform(0, 1.0, 3),
            color=np.random.uniform(0, 1.0, 4),
        )
mojo.close_passive_viewer()
