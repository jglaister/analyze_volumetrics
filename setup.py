#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
setup

Setup for nipypeVBM package

Author: Jeffrey Glaister
"""
from glob import glob
from setuptools import setup, find_packages

args = dict(
    name='analyze_volumetrics',
    version='0.1',
    description='Runs fslvbm as a Nipype pipeline',
    author='Jeffrey Glaister',
    author_email='jeff.glaister@gmail.com',
    url='https://github.com/jglaister/analyze_volumetrics'
)

setup(install_requires=['numpy', 'nibabel', 'pandas'],
      packages=['analyze_volumetrics'],
      **args)

#scripts=glob('bin/*'),

