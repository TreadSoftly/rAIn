#!/usr/bin/env python3
"""
Bootstrap the Argos CDK application.

Nothing to configure here - stack level settings live in argos_stack.py.
"""
from aws_cdk import App
from argos_stack import ArgosStack  # type: ignore[import-untyped]

app = App()
ArgosStack(app, "ArgosStack")
app.synth()
