#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
setup

Setup for analyze_volumetrics package

Author: Jeffrey Glaister
"""
from glob import glob
from setuptools import setup, find_packages

args = dict(
    name='analyze_volumetrics',
    version='0.1',
    description='Compiles and analyzes MACRUISE output volumetrics',
    author='Jeffrey Glaister',
    author_email='jeff.glaister@gmail.com',
    url='https://github.com/jglaister/analyze_volumetrics'
)

setup(install_requires=['numpy', 'nibabel', 'pandas', 'matplotlib'],
      packages=['analyze_volumetrics'],
      **args)
