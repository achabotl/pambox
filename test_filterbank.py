import pytest
import filterbank
import numpy as np
import scipy.io as sio


@pytest.mark.xfail
def test_third_octave_filtering_of_noise_():
    """There is a bug in the Matlab sEPSM code, the noise is filtered using
    the frequencies instead of the indexes, so this test cannot pass perfectly."""
    mat = sio.loadmat('./test_files/test_third_oct_filtered_noise.mat')
    noise = mat['noise'].squeeze()
    target_noise_spectrum = mat['noise_output_specs'].squeeze()
    target_noise_time = mat['noise_time_output'].squeeze().T
    center_f = mat['fcs1'].squeeze()
    noise_times, noise_spectra = \
        filterbank.noctave_filtering(noise, center_f, 22050, width=3)
    noise_rms = [general.rms(x) for x in noise_times]
    target_noise_rms = [general.rms(x) for x in target_noise_time]



def test_mod_filtering_for_simple_signal():
    signal = np.array([1, 0, 1, 0, 1])
    fs = 2205
    modf = np.array([1., 2., 4., 8., 16., 32., 64.])
    p = filterbank.mod_filterbank(signal, fs, modf)
    target = np.array([6.69785298e-18, 6.06375859e-06, 2.42555385e-05,
                      9.70302212e-05, 3.88249957e-04, 1.55506496e-03,
                      6.25329663e-03])
    np.testing.assert_allclose(p, target, rtol=1e-2)
