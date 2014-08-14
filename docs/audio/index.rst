Audio
=====

The :mod:`~pambox.audio` module provides a single function,
:py:func:`~pambox.audio.play`. By default, the output is scaled to
prevent clipping.

::

    from pambox import audio
    import numpy as np
    audio.play(np.random.randn(10000))


API
---

.. automodule:: pambox.audio
   :members:
