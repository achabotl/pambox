# -*- coding: utf-8 -*-
"""Processes performed by the 'central' auditory system.

Classes
-------

- ``IdealObs`` -- An IdealObserver, as used by [jorgensen2011]_.
- ``EPSMModulationFilterbank`` -- EPSM modulation filterbank, as used by [jorgensen2011]_.


"""
from __future__ import absolute_import

from .decision_metrics import IdealObs
from .ec import EC
from .modulation_filterbanks import EPSMModulationFilterbank

__all__ = (
    'EC',
    'EPSMModulationFilterbank'
    'IdealObs',
)

