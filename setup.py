from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

setup(
 ext_modules=cythonize([Extension("eos", ["eos.pyx"], include_dirs=[numpy.get_include()])]),
 zip_safe=False 
)
