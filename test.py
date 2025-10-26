import os
os.environ["QT_QPA_PLATFORM"] = "windows"
from PyQt5 import QtCore
print(QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.PluginsPath))

from pyvistaqt import BackgroundPlotter
import pyvista as pv

sphere = pv.Sphere()
pl = BackgroundPlotter(show=True)
pl.add_mesh(sphere)

def callback():
    print("tick")

pl.add_timer_event(max_steps=200, duration=500, callback=callback)
pl.show()
