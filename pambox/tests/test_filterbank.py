from __future__ import division, print_function
import pytest
from pambox import filterbank
import numpy as np
import scipy.io as sio
from numpy.testing import assert_allclose
from config import DATA_ROOT


def test_third_octave_filtering_of_noise_():
    mat = sio.loadmat(DATA_ROOT + '/test_third_oct_filt_rms.mat')
    noise = mat['x'].squeeze()
    target_noise_rms = mat['rms_out'].squeeze()
    center_f = mat['midfreq'].squeeze()
    rms_out = filterbank.noctave_filtering(noise, center_f, 22050, width=3)
    assert_allclose(target_noise_rms, rms_out, rtol=1e-4)


def test_mod_filtering_for_simple_signal():
    signal = np.asarray([1, 0, 1, 0, 1])
    fs = 2205
    modf = np.asarray([1., 2., 4., 8., 16., 32., 64.])
    p, _ = filterbank.mod_filterbank(signal, fs, modf)
    target = np.asarray([6.69785298e-18, 6.06375859e-06, 2.42555385e-05,
                         9.70302212e-05, 3.88249957e-04, 1.55506496e-03,
                         6.25329663e-03])
    assert_allclose(p, target, rtol=1e-2)


def test_mod_filt_complex():
    """Test modulation filtering with actual speech and noise signals
    """
    mat = sio.loadmat(DATA_ROOT + '/test_mod_filtering.mat')
    x = mat['data'].squeeze()
    fs = mat['fs'].squeeze()
    modf = np.hstack((mat['fcut'].squeeze(), mat['fcs'].squeeze()))
    modf = modf.astype('float')
    target = mat['powers'].squeeze()
    powers, _ = filterbank.mod_filterbank(x, fs, modf)
    assert_allclose(powers, target)


@pytest.mark.slow
def test_mod_filt_sepsm_v1():
    """Test modulation filtering with actual speech and noise signals
    """
    mat = sio.loadmat(DATA_ROOT + '/test_modFbank_v1.mat')
    x = mat['Env'][:, 0].squeeze()
    fs = mat['fs'].squeeze()
    modf = mat['fcs_EPSM'].squeeze()
    modf = modf.astype('float')
    target = mat['ExcPtn'][:, 0].squeeze()
    powers, _ = filterbank.mod_filterbank(x, fs, modf)
    assert_allclose(powers, target)


def test_mod_filterbank_for_temporal_outpout():
    """Test modulation filterbank for its temporal output.

    Mostly used in the mr-sEPSM model, where the time output is needed to
    process the envelope power for different window lenghts.
    """
    mat = sio.loadmat(DATA_ROOT +
                      '/test_mrsepsm_mod_filterbank_temporal_outputs.mat',
                      squeeze_me=True)
    x = mat['Env']
    fs = mat['fs']
    modf = mat['fcs']
    print(modf)
    p, t = filterbank.mod_filterbank(x, fs, modf)
    assert_allclose(p, mat['Penv'])
    assert_allclose(t, mat['Filtered_Env'].T)
