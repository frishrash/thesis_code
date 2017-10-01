from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

from sklearn._build_utils import get_blas_info
cblas_libs, blas_info = get_blas_info()

extensions = [
	Extension("_exlasso", ["_exlasso.pyx"],
		include_dirs=[numpy.get_include()],
		#include_dirs=[numpy.get_include(), 'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.109\windows\mkl\include'],
		
		#library_dirs=['C:\Program Files\Anaconda2\Library\lib'],
		#library_dirs=['C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.109\windows\mkl\lib\intel64_win'],
		#libraries=['libopenblas']
		#libraries=[cblas_libs],
		#extra_compile_args=blas_info.pop(
        #                     'extra_compile_args', []),
		#libraries=['mkl_intel_thread']
	)
]

setup(
    name = "My hello app",
    ext_modules = cythonize(extensions),
)