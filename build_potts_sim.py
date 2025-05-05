from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'potts_sim',
        ['potts_sim.cpp'],
        include_dirs=[pybind11.get_include(), pybind11.get_include(user=True)],
        language='c++',
        extra_compile_args=[
            '-Ofast',
            '-march=native',    
            '-flto',            
            '-funroll-loops',   
            '-fno-math-errno',
            '-fno-signed-zeros',
            '-ffinite-math-only',
            '-funswitch-loops'
        ],
        extra_link_args=[]
    )
]

setup(
    name='potts_sim',
    ext_modules=ext_modules,
    zip_safe=False,
)
