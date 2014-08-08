.. pambox documentation master file, created by
   sphinx-quickstart on Wed Jan 22 17:15:54 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PAMBOX
======

`pambox <https://github.com/achabotl/pambox>`_ is a Python toolbox to
facilitate the development of auditory models, with a focus on speech
intelligibility prediction models.


Installing
----------

Install PAMBOX with::

    pip install pambox

Structure of the toolbox
------------------------

The structure of the toolbox is inspired by the auditory system. The classes
and functions are split between a "peripheral" and a "central" part. The
"peripheral" part contains, obviously, the "outer", "middle",
and "inner" modules. The "central" part is more general and contains the
modules and functions for central processes, without much order for now.

Contents
--------

.. toctree::
   :maxdepth: 2

   audio/index
   periph/index
   central/index
   speech/index
   distort/index
   utils/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

