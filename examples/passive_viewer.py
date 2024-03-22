"""Example showing:

 - Spawning of objects
 - Moving objects
 - Using passive viewer
 - Setting colors
"""
import time
from pathlib import Path

import numpy as np

from mojo import Mojo
from mojo.elements import Body, Geom

mojo = Mojo(str(Path(__file__).parent / "world.xml"), 0.02)
mojo.launch_viewer(passive=True)
body = Body.create(mojo)
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
