"""FastAPI sidecar that exposes hydrus1d / swms2d / richards3d to a GUI.

This subpackage is the Tauri GUI's backend. It runs as a local HTTP
server on a chosen port and lets the desktop frontend submit
simulations and stream results.
"""
__version__ = "0.1.0"
