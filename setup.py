#!/usr/bin/env python
from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name = "geointerpreter",
    version = "1.0.0",
    description = "A Python package to predict permeability and facies from log and core image data",
    url = "https://github.com/ese-msc-2023/ads-arcadia-reservoirrocks",
    author = "ADS project Team Reservoir Rocks",
    packages = ["geointerpreter"],
    install_requires = requirements,
)