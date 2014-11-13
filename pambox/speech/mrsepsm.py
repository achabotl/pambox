# -*- coding: utf-8 -*-
from __future__ import division
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from six.moves import zip
from pambox.speech.sepsm import Sepsm
from pambox import central
from pambox import inner


class MrSepsm(Sepsm):
    """Multi-resolution envelope power spectrum model (mr-sEPSM).

    Parameters
    ----------
    fs : int, optional, (Default value = 22050)
        Sampling frequency.
    cf : array_like, optional
        Center frequency of the cochlear filters.
    modf : array_like, optional (Default value = _default_modf)
        Center frequency of modulation filters.
    downsamp_factor : int, optional, (Default value = 10)
        Envelope downsampling factor. Simply used to make calculattion faster.
    noise_floor : float, optional, (Default value = 0.001)
        Value of the internal noise floor of the model. The default is -30 dB.
    snr_env_limit : float, optional, (Default value = 0.001)
        Lower limit of the SNRenv values. Default is -30 dB.
    snr_env_ceil : float, optional, (Default value = None)
        Upper limit of the SNRenv. No limit is applied if `None`.
    min_win : float, optional, (Default value = None)
        Minimal duration of the multi-resolution windows, in ms.
    name : string, optional, (Default value = 'MrSepsm')
        Name of the model.

    References
    ----------
    .. [jorgensen2013multi] S. Joergensen, S. D. Ewert, and T. Dau: A
        multi-resolution envelope-power based model for speech
        intelligibility. J Acoust Soc Am 134 (2013) 436--446.

    """

    # Default center frequencies of the modulation filterbank.
    _default_modf = (1., 2., 4., 8., 16., 32., 64., 128., 256.)

    def __init__(self, fs=22050, cf=Sepsm._default_center_cf,
                 modf=_default_modf,
                 downsamp_factor=10,
                 noise_floor=0.001,
                 snr_env_limit=0.001,
                 snr_env_ceil=None,
                 min_win=None,
                 name='MrSepsm'
                 ):
        Sepsm.__init__(self, fs, cf, modf, downsamp_factor, noise_floor,
                       snr_env_limit)
        self.min_win = min_win
        self.name = name
        self.snr_env_ceil = snr_env_ceil

    def _mr_env_powers(self, channel_envs, filtered_envs):
        """Calculates the envelope power in multi-resolution windows.

        Parameters
        ----------
        channel_env : ndarray
            Envelope of the peripheral channel. The shape should be (N_SIG,
            N_CHAN, N)
        filtered_envs : ndarray
            Filtered envelope. The shape should be (N_SIG, N_CHAN, N_MODF, N)

        Returns
        -------
        mr_env_powers : masked_array
            Multi-resolution envelope powers of shape (N_SIG, N_CHAN, N_MODF,
            N_SEG), where N_SEG is the maximum number of segments in the
            multi-resolution process, which is for the highest modulation
            center frequency. For a 1 sec sample, it is about 400 segments.
            Low modulation frequencies come first.

        """
        # Here we find the duration and the number of windows for each
        # modulation center frequency...
        len_env = filtered_envs.shape[-1]
        win_durations = 1. / np.asarray(self.modf, dtype='float')
        if self.min_win is not None:
            win_durations[win_durations < self.min_win] = self.min_win
        win_lengths = np.floor(win_durations * self.fs / self
                               .downsamp_factor).astype('int')
        n_segments = np.ceil(len_env / win_lengths).astype('int')

        # ... then we calculate the DC power used for the normalization.
        # We divide it by 2 such that a fully modulated signal has an
        # AC-power of 1...
        dc_power = np.mean(channel_envs, axis=-1) ** 2 / 2

        # ... then we create a masked array of zeros, where all entries are
        # hidden...
        mr_env_powers = np.ma.masked_all(
            (filtered_envs.shape[:-1] + (np.max(n_segments),))
        )

        # ... and start looping throug the input signals...
        for i_sig, sig_envs in enumerate(filtered_envs):
            # ... and the channels...
            for i_chan, chan_envs in enumerate(sig_envs):
                # ... and the modulation channels, zipped with the window
                # lengths and number of segments...
                for i_modf, (n_seg, win_length, env) in enumerate(
                        zip(n_segments, win_lengths, chan_envs)):
                    n_complete_seg = n_seg - 1
                    last_idx = int(n_complete_seg * win_length)
                    # ... we reshape to n_seg x win_length so that we can
                    # calculate the variance in a single operation...
                    tmp_env = env[:last_idx].reshape((-1, win_length))
                    # ... then we normalize the variance by N-1, like in MATLAB.
                    tmp_env_powers = np.var(tmp_env, axis=-1, ddof=1) / \
                                     dc_power[i_sig, i_chan]
                    # ... and treat the last segment independently, just in
                    # case it is not # complete, i.e. it is shorter than the
                    # window length.
                    tmp_env_powers_last = np.var(env[last_idx:], ddof=1) / \
                                          dc_power[i_sig, i_chan]
                    # ... then we save it in our masked_array...
                    mr_env_powers[i_sig, i_chan, i_modf, :n_complete_seg] = \
                        tmp_env_powers
                    mr_env_powers[i_sig, i_chan, i_modf, n_complete_seg] = \
                        tmp_env_powers_last
                    # ... and finally make these values visible through the
                    # mask.
                    mr_env_powers.mask[i_sig, i_chan, i_modf, :n_seg] = False

        return mr_env_powers

    @staticmethod
    def _time_average(mr_snr_env):
        """Averages the multi-resolution SNRenv over time.

        Parameters
        ----------
        mr_snr_env : ndarray

        Returns
        -------
        ndarray

        """
        return mr_snr_env.mean(axis=-1)

    def _mr_snr_env(self, p_mix, p_noise):
        """Calculates the multi-resolution SNRenv.

        Parameters
        ----------
        p_mix, p_noise : ndarrays
            Envelope power of the mixture and of the noise alone.

        Returns
        -------
        mr_snr_env : masked_array
            Multi-resolution `masked_array` of SNRenv.
        exc_ptns : list of masked_arrays
            Multi-resolution values of the mixture and of the noise alone.

        """

        # First we limit the noise such that it cannot exceed the mix,
        # since they exist at the same time...
        p_noise = np.minimum(p_noise, p_mix)

        # ... then we restrict the noise floor restricted to set value
        # reflecting an internal noise threshold...
        p_mix = np.maximum(p_mix, self.noise_floor)
        p_noise = np.maximum(p_noise, self.noise_floor)

        # ... and we calculate the snrenv according to Eq (2) in Joergensen
        # et al. (2013)...
        mr_snr_env = (p_mix - p_noise) / p_noise
        mr_snr_env = np.maximum(mr_snr_env, self.snr_env_limit)
        if self.snr_env_ceil is not None:
            mr_snr_env = np.minimum(mr_snr_env, self.snr_env_ceil)

        return mr_snr_env, [p_mix, p_noise]

    def predict(self, clean, mixture, noise):
        """Predicts intelligibility using the mr-sEPSM.

        Predicts the SNRenv value for the combination of mixture and noise
        alone.

        Parameters
        ----------
        clean, mixture, noise : ndarrays
            Single-dimension arrays for the clean speech, the mixture of the
            clean speech ans noise, and the noise alone.

        Returns
        -------
        dict
            Dictionary with the predictions by the model.

        Notes
        -----


        """
        signals = np.vstack((clean, mixture, noise))

        bands_above_thres_idx = self._find_bands_above_thres(mixture)
        channel_sigs = self._peripheral_filtering(signals)
        channel_envs = self._extract_env(channel_sigs)
        channel_envs = self._mod_sensitivity(channel_envs)
        filtered_envs, lt_exc_ptns = self._mod_filtering(channel_envs)
        mr_exc_ptns = self._mr_env_powers(channel_envs, filtered_envs)
        mr_snr_env_matrix, _ = self._mr_snr_env(
            mr_exc_ptns[-2], mr_exc_ptns[-1]
        )
        time_av_snr_env_matrix = self._time_average(mr_snr_env_matrix)

        lt_snr_env_matrix, _ = super(MrSepsm, self)._snr_env(*lt_exc_ptns[-2:])
        lt_snr_env = super(MrSepsm, self)._optimal_combination(
            lt_snr_env_matrix, bands_above_thres_idx
        )

        snr_env = self._optimal_combination(
            time_av_snr_env_matrix, bands_above_thres_idx
        )

        res = {
            'p': {
                'snr_env': snr_env,
                'lt_snr_env': lt_snr_env,
            },
            'snr_env_matrix': time_av_snr_env_matrix,

            # Output of what is essentially the sEPSM.
            'lt_snr_env_matrix': lt_snr_env_matrix,
            'lt_exc_ptns': lt_exc_ptns,

            'mr_snr_env_matrix': mr_snr_env_matrix,
            'mr_exc_ptns': mr_exc_ptns,

            'bands_above_thres_idx': bands_above_thres_idx
        }
        return res


    def _optimal_combination(self, snr_env, bands_above_thres_idx):
        """Combines SNRenv across audio and modulation channels.
        
        Only modulation channels below 1/4 of the audio center frequency are
        considered.

        Parameters
        ----------
        snr_env : ndarray
            Linear values of SNRenv
        bands_above_thres_idx : ndarray
            Index of audio channels above threshold.

        Returns
        -------
        float
            SNRenv value.

        """
        # Acceptable modulation frequencies
        ma = np.tile(np.asarray(self.modf), (len(self.cf), 1)) \
             >= np.asarray(self.cf)[:, np.newaxis]
        # Set the reject SNRenvs to 0. It's ok because we're just doing a
        # summation
        snr_env[ma] = 0.
        # Combine across modulation filters...
        snr_env = np.sqrt(np.sum(snr_env[bands_above_thres_idx] ** 2,
                                 axis=-1))
        # ... and peripheral filters.
        snr_env = np.sqrt(np.sum(snr_env ** 2))
        return snr_env

    @staticmethod
    def _plot_mr_matrix(mat, x=None, y=None, fig=None, subplot_pos=111):
        """

        Parameters
        ----------
        mat :
        x :
            (Default value = None)
        y : array_like
            (Default value = None)
        x : arra_like
            (Default value = None)
        fig :
            (Default value = None)
        subplot_pos :
            (Default value = 111)

        Returns
        -------

        """

        n_y, n_x = mat.shape
        if y is None:
            y = np.arange(n_y)

        max_mat = mat.max()
        bmap = plt.get_cmap('PuBu')

        if fig is None:
            fig = plt.figure()
        else:
            pass

        grid = ImageGrid(fig, subplot_pos,
                         nrows_ncols=(n_y, 1),
                         aspect=False,
                         share_all=False,
                         cbar_mode='single',
                         cbar_location='right',
                         cbar_size='0.5%',
                         cbar_pad=0.05)

        for ax, p, f in zip(grid, mat[::-1], y[::-1]):
            values = p.compressed()
            extent = (0, 1, 0, 1)
            im = ax.imshow(values[np.newaxis, :],
                           aspect='auto',
                           interpolation='none',
                           extent=extent,
                           vmax=max_mat,
                           cmap=bmap)
            ax.grid(False)
            ax.set_yticks([0.5])
            ax.set_yticklabels([f])
        return im

    def plot_mr_exc_ptns(self
                         , ptns
                         , dur=None
                         , db=True
                         , vmin=None
                         , vmax=None
                         , fig_subplt=None
                         , attr='exc_ptns'
                         , add_cbar=True
                         , add_ylabel=True
                         , title=None
                         ):
        """Plots multi-resolution representation of envelope powers.

        Parameters
        ----------
        ptns : dict
            Predictions from the model. Must have a `mr_snr_env_matrix`
            key.
        dur : bool
            Display dB values of the modulation power or SNRenv values. (Default: True.)
        vmax : float
            Maximum value of the colormap. If `None`,
            the data's maxium value is used. (Default: None)
        db : bool
            Plot the values in dB. (Default value = True)
        vmin : float
            Minimum value of the heatmap. The value will be infered from the
            data if `None`. (Default value = None)
        fig_subplt : tuple of (fig, axes)
            Matplotlib `figure` and `axes` objects where the data should be
            plotted. If `None` is provided, a new figures with the necessary
            axes will be created. (Default value = None)
        attr : string
            Key to query in the `ptns` dictionary. (Default value = 'exc_ptns')
        add_cbar : bool
            Add a colorbar to the figure. (Default value = True)
        add_ylabel : bool
            Add a y-label to the axis. (Default value = True)
        title : bool
            Add a title to the axis. (Default value = None)

        Returns
        -------

        """

        mf = self.modf

        if 'exc_ptns' in attr:
            cbar_label = 'Modulation power'
        else:
            cbar_label = 'SNRenv'

        if db:
            ptns = 10 * np.log10(ptns)
            cbar_label += ' [dB]'
        else:
            ptns = ptns
            cbar_label += ' [lin]'

        if vmax is None:
            vmax = ptns.max()
        if vmin is None:
            vmin = ptns.min()

        n_mf, n_win = ptns.shape

        if dur is None:
            dur = n_win / self.modf[-1]

        if fig_subplt is None:
            fig = plt.figure()
            subplt = 111
        else:
            fig, subplt = fig_subplt

        if add_cbar:
            cbar_dict = {
            'cbar_mode':'single',
            'cbar_location':'right',
            'cbar_size':'0.5%',
            'cbar_pad':0.05}
        else:
            cbar_dict = {}

        bmap = plt.get_cmap('PuBu')
        xlabel = "Time [s]"
        ylabel = "Modulation frequency [Hz]"
        grid = ImageGrid(fig, subplt,
                         nrows_ncols=(n_mf, 1),
                         aspect=False,
                         # axes_pad=0.05,
                         # add_all=True,
                         share_all=False,
                         **cbar_dict)

        for ax, p, f in zip(grid, ptns[::-1], mf[::-1]):
            try:
                values = p.compressed()
            except AttributeError:
                values = p
            extent = (0, 1, 0, 1)
            im = ax.imshow(values[np.newaxis, :],
                           aspect='auto',
                           interpolation='none',
                           extent=extent,
                           vmin=vmin,
                           vmax=vmax,
                           cmap=bmap)
            ax.grid(False)
            # ax.set_yticks([0.5])
            ax.set_yticklabels([f])

        if add_cbar:
            cbar = grid.cbar_axes[0].colorbar(im)
            cbar.ax.set_ylabel(cbar_label)

        xticks_labels = np.arange(0, dur, 0.2)
        xticks = np.linspace(0, 1, len(xticks_labels))
        grid[-1].set_xticks(xticks)
        grid[-1].set_xticklabels(xticks_labels)
        grid[-1].set_xlabel(xlabel)
        if title:
            grid[0].set_title(title, )
        if add_ylabel:
            grid[n_mf // 2].set_ylabel(ylabel, labelpad=20)

        # fig.text(0.05, 0.5, ylabel, va='center', rotation=90, size=11)
        return fig

