#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from setuptools import setup
from setuptools.command.test import test as TestCommand
import codecs
import os
import sys
import re


here = os.path.abspath(os.path.dirname(__file__))

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


long_description = read('README.md')


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)

setup(
    name='pambox',
    version=find_version('pambox', '__init__.py'),
    url='https://bitbucket.org/achabotl/pambox',
    license='Modified BSD License',
    author='Alexandre Chabot-Leclerc',
    tests_require=['pytest'],
    install_requires=[],
    cmdclass={'test': PyTest},
    author_email='pambox@alex.alexchabot.net',
    description='A Python toolbox for auditory modeling',
    long_description=long_description,
    packages=['pambox'],
    include_package_data=True,
    platforms='any',
    test_suite='pambox.test.test_pambox',
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Operating System :: OS Independent'
    ],
    extras_require={
        'testing': ['pytest']
    }
)
