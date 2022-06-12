# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:12:16 2020

@author: katerina
"""
import setuptools  # noqa: F401 for Windows
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

ext_modules = [
        Extension('cython_functions',
                  ['cython_functions.pyx'],
                  include_dirs=[numpy.get_include()],
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
        ]


setup(
      name='Sample app',
      cmdclass={'build_ext': build_ext},
      ext_modules=cythonize(ext_modules, language_level=3)
      )
