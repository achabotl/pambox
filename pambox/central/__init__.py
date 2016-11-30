# -*- coding: utf-8 -*-
""":mod:`~pambox.central` contains processes performed by the 'central' auditory system.

Classes
-------

- :py:class:`~EC` -- An Equalization--Cancellation stage, as used by [wan2014]_.
- :py:class:`~EPSMModulationFilterbank` -- EPSM modulation filterbank, as used by [jorgensen2011]_.
- :py:class:`~IdealObs` -- An IdealObserver, as used by [jorgensen2011]_.


"""
from __future__ import absolute_import, division, print_function

__all__ = (
    'EC',
    'EPSMModulationFilterbank',
    'IdealObs',
)

from .decision_metrics import IdealObs
from .ec import EC
from .modulation_filterbanks import EPSMModulationFilterbank

