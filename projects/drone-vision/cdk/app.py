#!/usr/bin/env python3
"""
Bootstrap the Drone-Vision CDK application.

Nothing to configure here - stack level settings live in drone_stack.py.
"""
from aws_cdk import App
from drone_stack import DroneStack  # type: ignore[import-untyped]

app = App()
DroneStack(app, "DroneVisionStack")
app.synth()
