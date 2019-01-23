#!/usr/bin/env python3
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

scripts = ['kt.pyx'] 
for script in scripts:
    short = script.split('.')[0]
    setup(
        name=short, 
        ext_modules=cythonize(
            Extension(short, [script], include_dirs=[numpy.get_include()])),
        install_requires=['numpy', 'scipy', 'cython', 'pandas', 'progressbar2', 'matplotlib'])
