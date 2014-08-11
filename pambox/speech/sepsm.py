# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from six.moves import zip
from pambox import utils
from pambox import inner
try:
    import seaborn
except ImportError:
    pass


class Sepsm(object):
    """Implement the sEPSM intelligibility model

    Parameters
    ----------
    fs : int
         (Default value = 22050)
    cf : array_like
         (Default value = _default_center_cf)
    modf : array_like
         (Default value = _default_modf)
    downsamp_factor : int
         (Default value = 10)
    noise_floor : float
         (Default value = 0.01)
    snr_env_limit : float
         (Default value = 0.001)
    """

    _default_center_cf = (63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
                          800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
                          6300, 8000)
    _default_ht_diffuse = (37.5, 31.5, 26.5, 22.1, 17.9, 14.4, 11.4, 8.4, 5.8,
                           3.8, 2.1, 1.0, 0.8, 1.9, 0.5, -1.5, -3.1, -4.0,
                           -3.8, -1.8, 2.5, 6.8)
    _default_modf = (1., 2., 4., 8., 16., 32., 64.)

    def __init__(self, fs=22050
                 , cf=_default_center_cf
                 , modf=_default_modf
                 , downsamp_factor=10
                 , noise_floor=0.01
                 , snr_env_limit=0.001
                 ):
        self.fs = fs
        self.cf = cf
        self.modf = modf
        self.downsamp_factor = downsamp_factor
        self.noise_floor = noise_floor
        self.snr_env_limit = snr_env_limit
        self.ht_diffuse = self._default_ht_diffuse
        self.name = 'Sepsm'

    def _peripheral_filtering(self, signal, center_f):
        """

        Parameters
        ----------
        signal : ndarray
            Signal to filter.
        center_f : ndarray
            Center frequencies of the peripheral filters.

        Returns
        -------

        """
        g = inner.GammatoneFilterbank(center_f, self.fs)
        y = g.filter(signal)
        return y

    def _bands_above_thres(self, x):
        """Select bands above threshold

        Parameters
        ----------
        x : array_like,
            RMS value of each peripheral channel.

        Returns
        -------
        ndarray
            Index of the bands above threshold.

        
        """
        noise_rms_db = 20 * np.log10(x)
        # convert to spectrum level according to SII - ANSI 1997
        noise_spec_level_corr = noise_rms_db \
            - 10.0 * np.log10(sp.asarray(self.cf) * 0.231)
        max_idx = min(len(noise_spec_level_corr), len(self.ht_diffuse))
        b = noise_spec_level_corr[:max_idx] > self.ht_diffuse[:max_idx]
        idx = np.arange(len(noise_rms_db))
        return idx[b]

    def _snr_env(self, p_mix, p_noise):
        """Calculates SNR_env for a signal mixture and a noise.

        Parameters
        ----------
        p_mix, p_noise : ndarray
            Channel envelopes for the clean speech, mixture and noise alone,
            in that order
        Returns
        -------
        ndarray
            Linear values of SNRenv.

        """

        p_mix = np.asarray(p_mix)
        p_noise = np.asarray(p_noise)

        # set nan values to zero
        p_mix[np.isnan(p_mix)] = 0
        p_noise[np.isnan(p_noise)] = 0

        # noisefloor cannot exceed the mix, since they exist at the same time
        p_noise = np.minimum(p_noise, p_mix)

        # the noisefloor restricted to minimum 0.01 reflecting and internal
        # noise threshold
        p_mix = np.maximum(p_mix, self.noise_floor)
        p_noise = np.maximum(p_noise, self.noise_floor)

        # calculation of snrenv
        snr_env = (p_mix - p_noise) / p_noise
        snr_env = np.maximum(snr_env, self.snr_env_limit)

        return snr_env

    def _optimal_combination(self, snr_env, bands_above_thres_idx):
        """Calculates "optimal combination" of SNRenv above threshold.

        Parameters
        ----------
        snr_env : ndarray
            Linear values of SNRenv.
        bands_above_thres_idx : ndarray
            Index values of the bands above threshold

        Returns
        -------
        snr_env : float


        Notes
        -----
        Combines the SNR values as:

        .. math::

            \srqt(\sum_idx SNRenv_idx ^ 2)

        """
        snr_env = np.sqrt(np.sum(snr_env[bands_above_thres_idx] ** 2,
                                 axis=-1))
        snr_env = np.sqrt(np.sum(snr_env ** 2))
        return snr_env

    def _env_extraction(self, signal):
        """Extracts the envelope of a time signal.

        Parameters
        ----------
        signal : ndarray
            Input signal.

        Returns
        -------
        ndarray
            Low-pass filtered envelope.

        Notes
        -----
        The envelope is extracted by calculating the absolute value of the
        Hilbert transform, and then by low-pass filtering the envelope at a
        frequency of 150 Hz using a first order Butterworth filter.

        """
        # Extract envelope
        tmp_env = utils.hilbert_envelope(signal).squeeze()
        # Low-pass filtering
        tmp_env = inner.lowpass_env_filtering(tmp_env, 150.0,
                                                 n=1, fs=self.fs)
        # Downsample the envelope for faster processing
        return tmp_env[::self.downsamp_factor]

    def predict(self, clean, mixture, noise):
        """Predicts intelligibility
        
        Parameters
        ----------
        clean, mixture, noise : ndarrays
            Clean, mixture, and noise signals.

        Returns
        -------
        res : dict
            @todo

        """
        fs_new = self.fs / self.downsamp_factor
        n = len(clean)
        n_modf = len(self.modf)
        n_cf = len(self.cf)

        # find bands above threshold
        filtered_mix_rms = inner.noctave_filtering(mixture, self.cf,
                                                        self.fs, width=3)
        bands_above_thres_idx = self._bands_above_thres(filtered_mix_rms)

        exc_ptns = np.zeros((3, n_cf, n_modf))
        # For each band above threshold,
        # (actually, for debug purposes, maybe I should keep all...)
        for idx_band in bands_above_thres_idx:
            # Peripheral filtering, of just the band we process
            filtered_signals = [self._peripheral_filtering(signal,
                                                           self.cf[idx_band])
                                for signal in [clean, mixture, noise]]

            downsamp_env = \
                np.empty((3, np.ceil(n / self.downsamp_factor).astype('int')))
            for i_sig, signal in enumerate(filtered_signals):
                downsamp_env[i_sig] = self._env_extraction(signal)

                exc_ptns[i_sig, idx_band], _ = \
                    inner.mod_filterbank(downsamp_env[i_sig], fs_new,
                                              self.modf)

        # Calculate SNRenv
        snr_env_matrix = self._snr_env(*exc_ptns[-2:])

        snr_env = self._optimal_combination(snr_env_matrix,
                                            bands_above_thres_idx)

        res = {
            'p': {
                'snr_env': snr_env
            },
            'snr_env_matrix': snr_env_matrix,
            'exc_ptns': exc_ptns,
            'bands_above_thres_idx': bands_above_thres_idx
        }

        return res

    def plot_bands_above_thres(self, res):
        """Plot bands that were above threshold as a bar chart.
        
        Parameters
        ----------
        res : dict
            A `dict` as output by the sEPSM model. The dictionary must have a
            `bands_above_threshold_idx` key.

        Returns
        -------
        None

        """
        cf = self.cf
        cf_ticks = range(len(cf))
        bands = res.bands_above_thres_idx
        # Make the bar chart
        y = np.zeros_like(cf)
        y[bands] = 1

        f, ax = plt.subplots()
        ax.bar(cf_ticks, y, align='center')
        ax.set_xlim([-1, len(cf)])
        ax.set_xticks(cf_ticks[::3])
        ax.set_xticklabels(cf[::3])  # only octave-spaced values
        ax.set_yticks([])

        ax.set_xlabel('Center frequency [Hz]')

        return self

    def _plot_mod_matrix(self, mat, ax=None, vmin=None, vmax=None):
        """Plots a matrix of values values as a heat map.
        
        Parameters
        ----------
        mat : ndarray
            Modulation power or SNRenv values
        ax : axes, optional, (Default value = None)
            Axes where to plot. A new figure and plot will be created by
            default.
        vmin, vmax : float, optional, (Default value = None)
            Minimum and maximum value to normalize the color scale.

        Returns
        -------


        """
        cf = self.cf
        mf = self.modf

        xlabel = 'Modulation frequency [Hz]'
        ylabel = 'Channel frequency [Hz]'
        bmap = plt.get_cmap('PuBu')

        if ax is None:
            f, ax = plt.subplots()
            ax.set_xticks(range(len(mf))[::2])
            ax.set_xticklabels(mf[::2])
            ax.set_yticks(range(len(cf))[::3])
            ax.set_yticklabels(cf[::3])

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        im = ax.imshow(mat, origin='lower',
                       interpolation='none',
                       cmap=bmap,
                       vmin=vmin,
                       vmax=vmax,
                       aspect='auto')
        return im

    def plot_snr_env_matrix(self, res, ax=None, vmin=None, vmax=None):
        """

        Parameters
        ----------
        res : dict
            Output of the :py:func:`predict` function.
        ax : ax object
             Matplotlib axis where the data should be plotted. A new axis
             will be created if the value is `None`. (Default value = None)
        vmin : float, optional
             Minimum value of the heatmanp. The minimum value will be infered
             from the data if `None`. (Default value = None)
        vmax : float, optional
             Maxiumum value of the heatmanp. The maximum value will be infered
             from the data if `None`. (Default value = None)

        Returns
        -------

        """
        if not ax:
            fig, ax = plt.subplots()
        data = 10 * np.log10(res['snr_env_matrix'])
        data[np.isinf(data)] = np.nan
        if vmin is None:
            vmin = np.nanmin(data)
        if vmax is None:
            vmax = np.nanmax(data)
        im = self._plot_mod_matrix(data, ax=ax, vmin=vmin, vmax=vmax)
        cb = plt.colorbar(im)
        cb.set_label(r"SNRenv [dB]")
        ax.set_xlabel('Modulation frequency [Hz]')
        ax.set_ylabel('Channel frequency [Hz]')

        ax.set_xticks(range(len(self.modf))[::2])
        ax.set_xticklabels([int(x) for x in self.modf[::2]])
        ax.set_yticks(range(len(self.cf))[::3])
        ax.set_yticklabels(self.cf[::3])

        return ax

    def plot_exc_ptns(self, res, db=True, attr='exc_ptns', vmin=None,
                      vmax=None):
        """Plot the excitation patterns from a prediction.
        
        Parameters
        ----------
        res : dict
            Results from an sEPSM prediction. The dictionay should have a
            "exc_ptns" key. Otherwise, the key to use can be defined using the
            `attr` parameter.
        db : bool, optional, (Default value = `True`)
             Plot as dB if `True`, otherwise plots linear values.
        attr : string, optional, (Default value = 'exc_ptns')
             Dictionary key to use for plotting the excitation patters
        vmin, vmax : float, optional, (Default = None)
            Minimum and maximum value to normalize the color scale.

        Returns
        -------

        """
        cf = self.cf
        mf = self.modf
        if db:
            data = 10 * np.log10(res[attr])
        else:
            data = res['exc_ptns']
        if not vmax:
            vmax = data.max()
        if not vmin:
            vmin = np.maximum(-30, data.min())

        xlabel = 'Modulation frequency [Hz]'
        ylabel = 'Channel frequency [Hz]'
        titles = ['Clean', 'Mixture', 'Noise']
        n_subs = data.shape[0]

        fig = plt.figure(1, figsize=(10, 5), dpi=300)
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(1, n_subs),
                         axes_pad=0.2,
                         share_all=True,
                         cbar_location='right',
                         aspect=False,
                         cbar_mode='single',
                         cbar_size='7%',
                         cbar_pad=0.05)
        fig.subplots_adjust(wspace=0.05)

        for ax, exc_ptns in zip(grid, data):
            im = self._plot_mod_matrix(exc_ptns, ax=ax, vmin=vmin, vmax=vmax)

        for ax in grid:
            ax.set_xticks(range(len(mf))[::2])
            ax.set_xticklabels([int(x) for x in mf[::2]])
            ax.set_yticks(range(len(cf))[::3])
            ax.set_yticklabels(cf[::3])

        for ax, title in zip(grid, titles[-n_subs:]):
            ax.set_title(title)

        if 'exc_ptns' in attr:
            cbar_label = 'Modulation power [dB]'
        else:
            cbar_label = 'SNRenv [dB]'

        cbar = grid.cbar_axes[0].colorbar(im)
        cbar.set_label_text(cbar_label)

        grid[0].set_ylabel(ylabel)
        fig.text(0.5, 0.0, xlabel, ha='center', size=17.6)
        return self

    def plot_filtered_envs(self, envs, fs, axes=None):
        """Plot the filtered envelopes.

        Parameters
        ----------
        envs : ndarray
            List of envelope signals.
        fs : int
            Sampling frequency.
        axes : axes, (Default value = None)
             Matplotlib axes where to place the plot. Defaults to creating a
             new figure is `None`.

        Returns
        -------

        """
        mf = self.modf

        n_envs, len_env = envs.shape
        t = np.arange(len_env) / fs * 1000  # time in ms.
        fig, axes = plt.subplots(n_envs, 1, sharex=True, sharey=False)
        fig.subplots_adjust(hspace=0.05)

        for ax, env, f in zip(axes, envs[::-1], mf[::-1]):
            ax.plot(t, env)
            ax.set_yticks([env.mean()])
            ax.set_yticklabels([f])
            ax.set_xlim([t.min(), t.max()])

        axes[-1].set_xlabel('Time [ms]')
        fig.text(0.05, 0.5, 'Filter center frequency [Hz]',
                 va='center', rotation=90, size=11)
        return fig
