from setuptools import setup

setup(
  name='shapwrap',
  version='0.1.0',
  packages=['shapwrap'],
    author="Denis Kazakov",
    author_email="deka6994@colorado.edu",
    description="A wrapper for SHAP library. Makes experimenting faster and extends functionality. ",
    install_requires=[
        'shap'
    ],
)