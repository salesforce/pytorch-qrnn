from Cython.Build import cythonize
import numpy
import Cython.Compiler.Options
from distutils.core import setup
from distutils.extension import Extension
Cython.Compiler.Options.cimport_from_pyx = True

exts = [
    Extension(name='torchqrnn._cpu_forget_mult',
        sources=['torchqrnn/_cpu_forget_mult.pyx'],
        language='c')
]

setup(
    name='PyTorch-QRNN',
    version='0.1',
    packages=['torchqrnn',],
    license='BSD 3-Clause License',
    long_description=open('README.md').read(),
    ext_modules=cythonize(exts)
)
