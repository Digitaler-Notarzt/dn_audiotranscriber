from setuptools import setup, find_packages

setup(
    name='dn_audiotranscriber',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'sounddevice',
        'scipy',
        'openai-whisper',
        'numpy'
    ],
)