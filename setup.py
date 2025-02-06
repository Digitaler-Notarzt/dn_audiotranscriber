from setuptools import setup, find_packages
import importlib.util

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

def read_version(fname="dn_audiotranscriber/__version__.py"):
    exec(compile(open(fname, encoding="utf-8").read(), fname, "exec"))
    return locals()["__version__"]

setup(
    name='dn_audiotranscriber',
    version="0.0.11",
    packages=find_packages(),
    install_requires=requirements,
)
