# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sepsm
import numpy as np
import scipy as sp 
import scipy.signal
import matplotlib.pyplot as plt
import scipy.io as sio
%pdb off
import time
import filterbank
from matplotlib.collections import LineCollection

# <codecell>

from auditory import gammatonefir

# <codecell>

mat = sio.loadmat("./test_files/test_gammatone_filtering.mat")
midfreq = mat['midfreq'][0].astype('float')
fs = mat['fs'].astype('float')
signal = mat['signal'].T.astype('float')
## Target, first index is for speech only.
target = mat['GT_output'][:, :, 0].T
plot(signal)

# <markdowncell>

# ## Filtering

# <markdowncell>

# Create a filter of the same length as the signal so that I don't have to do the middle padding myself. It would probably be a better idea to do so, though. I could port `middlepad` from LTFAT. 
# 
# The filter length has to be even.

# <codecell>

b = gammatonefir(midfreq, np.array(22050.)) 
print(b.shape)

# <markdowncell>

# The filter now has zeroes in the middle and has the right length.

# <codecell>

plot(np.real(b[0]))

# <markdowncell>

# Now if we look at the spectrum of that filter:

# <codecell>

log_spectrum = 20.*sp.log10(np.abs(sp.fftpack.fft(b[10])))
plot(sp.linspace(0,11025,5000), log_spectrum)

# <codecell>

filterbank.mfreqz(b[10], 1.0, 22050.)

# <markdowncell>

# Now if we use the real output of the function.

# <codecell>

b_real = gammatonefir(midfreq, np.array(22050.), ftype='real') 
b_real_shifted = sp.fftpack.fftshift(b_real[10])
plot(b_real_shifted[2300:])

# <codecell>

b_real_spectrum = sp.fftpack.fft(b_real_shifted[2300:])
plot(b_real_spectrum)

# <markdowncell>

# Do the actual filtering.

# <codecell>

tstart = time.time()
y = np.array([2 * np.real(sp.signal.fftconvolve(signal.flat, den, mode='full')[0:len(signal)]) for den in b])
print(time.time() - tstart)

# <codecell>

tstart = time.time()
y2 = np.empty((b.shape[0], signal.shape[0]))
for ii, den in enumerate(b):
    y2[ii] = np.real(sp.signal.fftconvolve(signal.flat, den, mode='full')[0:len(signal)].flat)
print(time.time() - tstart)

# <codecell>

print(y.shape)
subplot(211)
plot(y[1])
subplot(212)
plot(y[21])

# <markdowncell>

# ## Testing of equality with MATLAB implementation

# <codecell>

np.testing.assert_allclose(y, target, atol=1e-5)

# <codecell>

f, axarr = plt.subplots(4,4, sharex=True, sharey=True)
for ii in np.arange(16):
    axarr[ii % 4, floor(ii/4)].plot(target[ii, ::100])
    axarr[ii % 4, floor(ii/4)].plot(y[ii, ::100])
axarr[0,0].set_ylim([-0.1, 0.1])
axarr[0,0].set_xlim([0, 350])
plt.legend(('target','y'))
f.set_size_inches(12,10)

# <markdowncell>

# ## Test effect of filter length

# <codecell>

b = gammatonefir(np.array([250.]), 22050., n=len(signal)+1)
print(b.shape)

# <codecell>

y2 = np.empty((b.shape[0], signal.shape[0]))
for ii, den in enumerate(b):
    y2[ii] = np.real(sp.signal.fftconvolve(signal.flat, den, mode='full')[0:len(signal)].flat)
    
plot(y2)
plot(target[6])

# <markdowncell>

# # Test the GammatoneApply code

# <codecell>

import gammatone as gt

# <codecell>

b_gtm, a_gtm, _, _, _ = gt.GammaToneMake(22050, midfreq)

# <codecell>

filterbank.mfreqz(b_gtm[10], a_gtm[10], 22050.)

# <codecell>

tstart = time.time()
y_gtm = gt.GammaToneApply(signal.squeeze(), b_gtm, a_gtm)
print(time.time() - tstart)

# <markdowncell>

# This implementation is definitely faster than using the gammatonefir implementation. It's probably because it don't know how to use the other one correctly...

# <codecell>

from general import rms

# <codecell>

rms_data = sp.zeros(16)
rms_gtm = rms_data.copy()
for ii in np.arange(16):
    rms_data[ii] = rms(target[ii])
    rms_gtm[ii] = rms(y_gtm[ii])
plot(rms_data)
plot(rms_gtm)
legend(('Data','Gtm'))
xlabel('Frequency bin')
ylabel('RMS')

# <markdowncell>

# The per-band RMS is different for the `target` and `filtered_speech`. It seems to be about a factor of 2.

# <codecell>

plot(sp.true_divide(rms_data, rms_gtm))
ylim((0,6))
xlabel('Frequency bin')
ylabel('Ratio between data and gtm RMS')

# <markdowncell>

# Right, it's a factor of two. I don't know where it comes from... 
# 
# Now plot all the bands from the target and the filtered speech, with the scaling of two.

# <codecell>

f, axarr = plt.subplots(4,4, sharex=True, sharey=True)
for ii in np.arange(16):
    axarr[ii % 4, floor(ii/4)].plot(target[ii, ::100],'b')
    axarr[ii % 4, floor(ii/4)].plot(2 * y_gtm[ii, ::100],'g')
axarr[0,0].set_ylim([-0.1, 0.1])
axarr[0,0].set_xlim([0, 350])
plt.legend(('target','y'))
f.set_size_inches(12,10)

# <markdowncell>

# Looks pretty good. There's a small difference in delay, but probably nothing major, especially since we're mostly interested in the magnitude response of the filters.

# <codecell>

o = np.ones(fs)
x = np.arange(fs) / fs.squeeze()
imp_resp_gtm = gt.GammaToneApply(o, b_gtm, a_gtm)

# <codecell>

# We need to set the plot limits, they will not autoscale
ax = axes()
ax.set_xlim((0, 1))
ax.set_ylim((np.min(np.min(imp_resp_gtm)),np.max(amax(imp_resp_gtm))))

line_segments = LineCollection([list(zip(x,y)) for y in imp_resp_gtm])
line_segments.set_array(x)
ax.add_collection(line_segments)
fig = gcf()
ax.set_xlim([0,0.2])

# <codecell>

plot(x,imp_resp_gtm[12])
xlim((0,0.03))
xlabel('Time [s]')

# <markdowncell>

# Now if we look directly at the impulse response:

# <codecell>

len = 22050
f0 = midfreq[12]  # 1 kHz
b = 130

# <codecell>

a = -2*pi*b
t = np.arange(len)/len
g = t ** 3 * np.exp(a*t) * np.cos(2*pi*f0*t)
g = 1./scale * g

# <codecell>

plot(t,g)
xlim((0, 0.03))

# <markdowncell>

# All right... they're not _exactly_ the same, but they have the same length. The scaling is completely off though.

# <markdowncell>

# # Filtering of delta pulse

# <markdowncell>

# Mostly to check the scaling.

# <codecell>

fs = 22050.0
delta = np.zeros(fs)
delta[0] = 1
f0 = midfreq  # 1 kHz

# <codecell>

b_gtm, a_gtm, _, _, _ = gt.GammaToneMake(22050, f0)
y2 = gt.GammaToneApply(delta, b_gtm, a_gtm)

# <codecell>

plot(y2[12,0:1000])

# <markdowncell>

# I just checked the output using the following Matlab code and indeed, for some reason, the impulse response in Python is _half_ the impulse response in Matlab.

# <markdowncell>

# ```matlab 
# x = zeroes(22050, 1);
# x(1) = 1;
# g = gammatonefir(1000, 22050, 'complex') 
# tmp = 2 * real(ufilterbank(x, g, 1));
# plot(tmp(1:1000))
# ```

# <markdowncell>

# !! Here's the factor two! Seriously, why do that?

