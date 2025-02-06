from setuptools import setup, find_packages
import importlib.util

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='dn_audiotranscriber',
    version="0.0.11",
    packages=find_packages(),
    install_requires=requirements,
)
