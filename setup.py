#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from setuptools import setup
from setuptools.command.test import test as TestCommand
import codecs
import os
import re


here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    # intentionally *not* adding an encoding option to open
    return codecs.open(os.path.join(here, *parts), 'r').read()

def read(*parts):
    # intentionally *not* adding an encoding option to open
    return codecs.open(os.path.join(here, *parts), 'r').read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


long_description = read('README.rst')


def check_dependencies():

    # Just make sure dependencies exist, I haven't rigorously
    # tested what the minimal versions that will work are
    # (help on that would be awesome)
    try:
        import numpy
    except ImportError:
        raise ImportError("pambox requires numpy")
    try:
        import scipy
    except ImportError:
        raise ImportError("pambox requires scipy")
    try:
        import matplotlib
    except ImportError:
        raise ImportError("pambox requires matplotlib")
    try:
        import pandas
    except ImportError:
        raise ImportError("pambox requires pandas")


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ['--runslow', 'pambox/tests']
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


if __name__ == '__main__':

    import sys
    if not (len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
            sys.argv[1] in ('--help-commands', 'egg_info', '--version',
                            'clean'))):
        check_dependencies()

    setup(
        name='pambox',
        description='A Python toolbox for auditory modeling',
        author='Alexandre Chabot-Leclerc',
        author_email='pambox@alex.alexchabot.net',
        version=find_version('pambox', '__init__.py'),
        url='https://bitbucket.org/achabotl/pambox',
        license='Modified BSD License',
        tests_require=['pytest'],
        install_requires=[
            'six>=1.4.1',
        ],
        cmdclass={'test': PyTest},
        long_description=long_description,
        packages=['pambox'],
        include_package_data=True,
        platforms='any',
        test_suite='pambox.tests',
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Natural Language :: English',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.4',
            'Topic :: Scientific/Engineering',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS'
        ],
        extras_require={
            'testing': ['pytest']
        }
    )
