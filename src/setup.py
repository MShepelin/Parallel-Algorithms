#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import io
import re
import os
import platform
import pathlib
import struct
import shutil
from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig

DEPENDENCIES = ['numpy']
CURDIR = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(os.path.dirname(CURDIR), "README.md"), "r", encoding="utf-8") as f:
    README = f.read()

def read_text(file_name: str):
    return open(os.path.join("",file_name)).read()
        
class build_ext(build_ext_orig):

    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        
        cwd = pathlib.Path().absolute()

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing

        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        # example of cmake args
        config = 'Debug' if self.debug else 'Release'
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute()),
            '-DCMAKE_BUILD_TYPE=' + config,
            '-DCMAKE_CUDA_ARCHITECTURES=52'
        ]

        # example of build args
        build_args = [
            '--config', config
            #'--', '-j4'
        ]

        os.chdir(str(build_temp))
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args)
        # Troubleshooting: if fail on line above then delete all possible
        # temporary CMake files including "CMakeCache.txt" in top level dir.
        os.chdir(str(cwd))

        if platform.system() == 'Windows':
            dll_path = os.path.join(build_temp, 'Release', 'parallel_rank.dll')

            shutil.copy(dll_path, extdir.parent)

setup(
    name="parallelrank",
    version="0.1.0",
    author="Dmitrii Shepelin",
    author_email="shepelin.d@yandex.ru",
    description="Primitive library for parallel rank computation",
    long_description=README,
    long_description_content_type = 'text/markdown',
    url="none",

    ext_modules=[Extension('parallelrank', [])],

    packages=find_packages(),
    cmdclass=dict(build_ext=build_ext),
    zip_safe=False,
    install_requires=DEPENDENCIES,
   
    license="License :: OSI Approved :: MIT License",
    classifiers=[
        "Programming Language :: Python",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",     
        "Operating System :: MacOS :: MacOS X"
    ],
)