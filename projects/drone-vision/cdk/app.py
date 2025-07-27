#!/usr/bin/env python3
from aws_cdk import App
from drone_stack import DroneStack
app = App()
DroneStack(app, "DroneVisionStack")
app.synth()
