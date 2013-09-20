import pytest
import numpy as np
import scipy as sp
import scipy.io as sio
import distort


@pytest.fixture
def mat_overlap_and_add():
    return sio.loadmat('./test_files/test_overlap_and_add.mat')


def test_overlap_and_add(mat_overlap_and_add):
    powers = mat_overlap_and_add['XNEW'].squeeze().T
    phases = mat_overlap_and_add['yphase'].squeeze().T
    len_window = mat_overlap_and_add['windowLen'].squeeze()
    shift = mat_overlap_and_add['ShiftLen'][0].squeeze()
    target = mat_overlap_and_add['ReconstructedSignal'].squeeze()
    y = distort.overlap_and_add(powers, phases, len_window,  shift)
    np.testing.assert_allclose(y, target, atol=1e-15)

