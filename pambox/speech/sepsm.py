# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from six.moves import zip
from pambox import central
from pambox import inner
try:
    import seaborn
except ImportError:
    pass


class Sepsm(object):
    """Implement the sEPSM intelligibility model [1].

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

    Notes
    -----

    Modifed on 2014-10-07 by Alexandre Chabot-Leclerc: Remove the unnecessary
    factor to compensate for filter bandwidth when computing the bands above
    threshold. The diffuse hearing threshold are already adjusted for filter
    bandwidth.

    References
    ----------

    .. [1] S. Jørgensen and T. Dau: Predicting speech intelligibility based
        on the signal-to-noise envelope power ratio after modulation-
        frequency selective processing. J. Acoust. Soc. Am. 130 (2011)
        1475--1487.

    """

    # Center frequencies of the peripheral filterbank.
    _default_center_cf = (63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
                          800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
                          6300, 8000)
    # Diffuse field hearing threshold in quiet: ISO 389-7:2005
    _default_ht_diffuse = (37.5, 31.5, 26.5, 22.1, 17.9, 14.4, 11.4, 8.4, 5.8,
                           3.8, 2.1, 1.0, 0.8, 1.9, 0.5, -1.5, -3.1, -4.0,
                           -3.8, -1.8, 2.5, 6.8)
    # Center frequencies of the modulation filterbank.
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

    def _peripheral_filtering(self, signals):
        """Filters a time signal using a Gammatone filterbank.

        Parameters
        ----------
        signal : ndarray
            Signal to filter.
        center_f : ndarray
            Center frequencies of the peripheral filters.

        Returns
        -------
        y : ndarray
            Outputs of the peripheral filterbank.

        """
        y = np.zeros((signals.shape[0], len(self.cf), signals.shape[-1]))
        g = inner.GammatoneFilterbank(self.cf, self.fs)
        for i_sig, s in enumerate(signals):
            y[i_sig] = g.filter(s)
        return y

    def _bands_above_thres(self, x):
        """Select bands above threshold accoring to the diffuse field hearing
        threshold in quiet: ISO 389-7:2005.

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
            - 10.0 * np.log10(sp.asarray(self.cf))
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

        p_mix = np.asanyarray(p_mix)
        p_noise = np.asanyarray(p_noise)

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

        return snr_env, (p_mix, p_noise)

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

    def _find_bands_above_thres(self, mixture):
        """Find the indexes of the bands that are above hearing threshold.

        The signal is filtered using a rectangular third-octave filterbank
        and the level in each bands is compared to the diffuse field hearing
        threshold in quiet: ISO 389-7:2005.

        Parameters
        ----------
        mixture : ndarray
            1D time signal.

        Returns
        -------
        indexes : ndarray
            Index of the channels that are above hearing threshold.

        See also
        --------
        _bands_above_thres : Compares the band powers to the hearing
            threshold.

        """
        filtered_rms_mix = inner.noctave_filtering(mixture, self.cf,
                                                   self.fs, width=3)
        return self._bands_above_thres(filtered_rms_mix)

    def _extract_env(self, channel_sigs):
        """Calculates the Hilbert envelope.

        Parameters
        ----------
        channel_sigs : ndarray
            Peripheral subband signals.

        Returns
        -------
        env : ndarray
            Hilbert envelope of the input signals.

        See also:
        ---------
        inner.hilbert_envelope : Calculates the Hilbert envelope.

        """
        return inner.hilbert_envelope(channel_sigs)

    def _mod_sensitivity(self, envs):
        """Reduces modulation sensitivity using a low-pass filter.

        Low-pass filters the envelope using a 1st-order Butterworth filter at
        150 Hz [1, 2].

        Parameters
        ----------
        envs : ndarray
            Envelopes

        Returns
        -------
        envs : ndarray

        References
        ----------
        .. [1] S. D. Ewert and T. Dau: Characterizing frequency selectivity
            for envelope fluctuations.. J. Acoust. Soc. Am. 108 (2000)
            1181-1196.
        .. [2] A. Kohlrausch, R. Fassel, and T. Dau: The influence of carrier
            level and frequency on modulation and beat-detection thresholds
            for sinusoidal carriers. J. Acoust. Soc. Am. 108 (2000) 723-734.



        """
        return inner.lowpass_env_filtering(envs, 150.0, n=1, fs=self.fs)

    def _mod_filtering(self, channel_envs):
        """Filters the subband envelopes using a modulation filterbank.

        Parameters
        ----------
        channels_envs : ndarray
            Subband envelopes. The shape should be (N_SIG, N_CHAN, N).

        Returns
        -------
        envs : ndarray
            Modulation subband signals. The shape is (N_SIG, N_CHAN, N_MODF, N).
        powers : ndarray
            Modulation power at the output of the modulation filterbank. The
            shape is (N_SIG, N_CHAN, N_MODF).

        """
        fs_new = self.fs / self.downsamp_factor
        # Downsample the envelope for faster processing
        channel_envs = channel_envs[..., ::self.downsamp_factor]
        if (channel_envs.shape[-1] % 2) == 0:
            len_offset = 1
        else:
            len_offset = 0
        envs = np.zeros((channel_envs.shape[0],
                         len(self.cf),
                         len(self.modf),
                         channel_envs.shape[-1] - len_offset)
        )
        powers = np.zeros((channel_envs.shape[0],
                         len(self.cf),
                         len(self.modf))
        )
        for i_sig, s in enumerate(channel_envs):
            for i_chan, chan in enumerate(s):
                powers[i_sig, i_chan], envs[i_sig, i_chan] =  \
                    central.mod_filterbank(chan, fs_new, self.modf)
        return envs, powers

    def predict(self, clean, mixture, noise):
        """Predicts intelligibility
        
        Parameters
        ----------
        clean, mixture, noise : ndarrays
            Clean, mixture, and noise signals.

        Returns
        -------
        res : dict
            Dictionary of the model predictions. The keys are as follow:
            - 'p': is a dictionary with the model predictions. In this case
            it contains a 'snr_env' key.
            - 'snr_env_matrix': 2D matrix of the SNRenv as a function audio
            frequency and modulation frequency.
            - 'exc_ptns': Modulation power at the output of the modulation
            filterbank for the intput signals. It is a (N_SIG, N_CHAN,
            N_MODF) array.
            - 'band_above_thres_idx': Array of the indexes of the bands that
            were above hearing threshold.

        """
        signals = np.vstack((clean, mixture, noise))

        # find bands above threshold
        bands_above_thres_idx = self._find_bands_above_thres(mixture)

        channel_sigs = self._peripheral_filtering(signals)
        channel_envs = self._extract_env(channel_sigs)
        channel_envs = self._mod_sensitivity(channel_envs)
        filtered_envs, exc_ptns = self._mod_filtering(channel_envs)
        snr_env_matrix, _ = self._snr_env(*exc_ptns[-2:])
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
        cf_ticks = list(range(len(cf)))
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
            ax.set_xticks(list(range(len(mf)))[::2])
            ax.set_xticklabels(mf[::2])
            ax.set_yticks(list(range(len(cf)))[::3])
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

        ax.set_xticks(list(range(len(self.modf)))[::2])
        ax.set_xticklabels([int(x) for x in self.modf[::2]])
        ax.set_yticks(list(range(len(self.cf)))[::3])
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
            ax.set_xticks(list(range(len(mf)))[::2])
            ax.set_xticklabels([int(x) for x in mf[::2]])
            ax.set_yticks(list(range(len(cf)))[::3])
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
