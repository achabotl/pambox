Python Auditory Modeling Toolbox
================================

.. image:: https://travis-ci.org/achabotl/pambox.svg?branch=develop
    :target: https://travis-ci.org/achabotl/pambox
.. image:: http://pambox.readthedocs.io/en/latest/?badge=latest
    :target: http://pambox.readthedocs.org

pambox is a Python toolbox to facilitate the development of auditory
models, with a focus on speech intelligibility prediction models.

The project is maintained by `@AlexChabotL <https://twitter.com/AlexChabotL>`_.

pambox provides a consistent API for speech intelligibility models,
inspired by `Scikit-learn <http://scikit-learn.org/>`_, to facilitate
comparisons across models.

Links:
~~~~~~

-  Official source code repo: https://github.com/achabotl/pambox
-  HTML documentations: http://pambox.readthedocs.org
-  Issue tracker: https://github.com/achabotl/pambox/issues
-  Mailing list: python-pambox@googlegroups.com
-  Mailing list archive: https://groups.google.com/d/forum/python-pambox

Dependencies
------------

pambox is tested to work under Python 2.7 and Python 3.4 (thanks to
``six``). Only Mac OS X (10.9) has been tested thoroughly.

The main dependencies are :

- `Numpy <http://www.numpy.org/>`_ >= 1.8.0,
- `Scipy <http://scipy.org/scipylib/>`_ >=0.14.0,
- `Pandas <http://pandas.pydata.org>`_ >=0.14.1,
- `six <https://bitbucket.org/gutworth/six>`_ >=1.7.2 (to have a single
  codebase for Python 2 and Python 3).
- `ipython-notebook <http://ipython.org>`_ >= 2.3.1 (for parallel experiments)

Lower versions of these packages are likely to work as well but have not been
thoroughly tested.

`pyaudio <http://people.csail.mit.edu/hubert/pyaudio/>`_ is required if you
want to use the ``audio`` module.

For running tests, you will need `pytest <http://pytest.org/>`_ and `pytest-cov <https://pypi.python.org/pypi/pytest-cov>`_.

Install
-------

Right now, `pambox` is only avaible through Github. It should be available
via `pip` soon. To install pambox from source::

    git clone https://github.com/achabotl/pambox.git
    cd pambox
    python setup.py install

If you need more details, see the
[Installation](https://github.com/achabotl/pambox/wiki/Installation) page on
the wiki.


Contributing
------------

You can check out the latest source and install it for development with:

::

    git clone https://github.com/achabotl/pambox.git
    cd pambox
    python setup.py develop

To run tests (you will need `pytest`), from the root pambox folder, type:

::

    python setup.py test

License
-------

pambox is licensed under the New BSD License (3-clause BSD license).
