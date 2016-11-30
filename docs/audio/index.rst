Audio
=====

The :mod:`~pambox.audio` module provides a single function,
:py:func:`~pambox.audio.play`. By default, the output is scaled to
prevent clipping and the sampling frequency is 44.1 KHz. Here a simple
example where we play some white noise created with NumPy::

    from pambox import audio
    import numpy as np
    audio.play(np.random.randn(10000))

To play back the signal without normalization, simply set ``normalize`` to
``False``. Be careful here! `It might get loud! <https://www.youtube.com/watch?v=Ul6xOdPCsCE>`_)::

    audio.play(np.random.randn(10000), normalize=False)


API
---

.. automodule:: pambox.audio
   :members:
