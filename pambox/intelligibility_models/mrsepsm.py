# -*- coding: utf-8 -*-
from __future__ import division
import brewer2mpl
import seaborn as sns
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from collections import namedtuple
from itertools import izip
from pambox.intelligibility_models.sepsm import Sepsm
from pambox import general
from pambox import filterbank
from pambox import auditory


class MrSepsm(Sepsm):
    """Docstring for MrSepsm. """

    _default_modf = (1., 2., 4., 8., 16., 32., 64., 128., 256.)

    def __init__(self, fs=22050, cf=Sepsm._default_center_cf,
                 modf=_default_modf,
                 downsamp_factor=10,
                 noise_floor=0.001,
                 snr_env_limit=0.001):
        """@todo: to be defined1. """
        Sepsm.__init__(self, fs, cf, modf, downsamp_factor, noise_floor,
                       snr_env_limit)

    def _mr_env_powers(self, channel_env, filtered_envs):
        len_env = filtered_envs.shape[-1]
        win_durations = 1. / np.asarray(self.modf, dtype='float')
        win_lengths = np.floor(win_durations * self.fs / self
                               .downsamp_factor).astype('int')
        n_segments = np.ceil(len_env / win_lengths).astype('int')

        # DC power used for normalization. Divided by 2 such that a fully
        # modulated signal has # an AC-power of 1.
        dc_power = np.mean(channel_env) ** 2 / 2

        # Create a masked array of zeros, where all entries are hidden.
        mr_env_powers = np.ma.masked_all((len(self.modf), max(n_segments)))

        for i_modf, (n_seg, env, win_length) in enumerate(
                izip(n_segments, filtered_envs, win_lengths)):
            n_complete_seg = n_seg - 1
            last_idx = int(n_complete_seg * win_length)
            # Reshape to n_seg x win_length so that we can calculate the
            # variance in a single operation
            tmp_env = env[:last_idx].reshape((-1, win_length))
            # Normalize the variance by N-1, like in MATLAB.
            tmp_env_powers = np.var(tmp_env, axis=-1, ddof=1) / dc_power
            # Treat the last segment independently, just in case it is not
            # complete, i.e. it is shorter than the window length.
            tmp_env_powers_last = np.var(env[last_idx:], ddof=1) / dc_power
            mr_env_powers[i_modf, :n_complete_seg] = tmp_env_powers
            mr_env_powers[i_modf, n_complete_seg] = tmp_env_powers_last

            mr_env_powers.mask[i_modf, :n_seg] = False

        return mr_env_powers

    def _time_average(self, mr_snr_env):
        return mr_snr_env.mean(axis=-1)

    def _mr_snr_env(self, p_mix, p_noise):
        """Calculate the multi-resolution SNRenv.

        :param mr_env_powers_mix:
        :param mr_env_powers_noise:
        :returns: tuple( ndarray, list), time-average of the mr-SNRenv, and mr-SNRenv.
        """

        # noisefloor cannot exceed the mix, since they exist at the same time
        p_noise = np.minimum(p_noise, p_mix)

        # the noise floor restricted to minimum 0.01 reflecting an
        # internal noise threshold
        p_mix = np.maximum(p_mix, self.noise_floor)
        p_noise = np.maximum(p_noise, self.noise_floor)

        # calculation of snrenv
        mr_snr_env = (p_mix - p_noise) / p_noise
        mr_snr_env = np.maximum(mr_snr_env, self.snr_env_limit)

        snr_env_matrix = self._time_average(mr_snr_env)

        return snr_env_matrix, [p_mix, p_noise], mr_snr_env


    def _mr_optimal_combination(self, powers, bands_above_thres_idx):
        """
        :powers: multi-resolution envelope modulation powers
        :mask: bands above threshold
        """
        snr_env = 0
        # Instead of looping over the channel dimensions and the modulation
        # channel dimensions, just flatten the list.
        for each in chain.from_iterable(powers):
            snr_env += each.sum() ** 2
        snr_env = np.sqrt(snr_env)
        return snr_env

    def predict(self, clean, mixture, noise):

        fs_new = self.fs / self.downsamp_factor
        N = len(clean)
        N_modf = len(self.modf)
        N_cf = len(self.cf)

        # Process only the mixture and noise if the clean speech is the same
        # as the noise.
        if (clean is None) or (np.array_equal(clean, mixture)):
            signals = (mixture, noise)
        else:
            signals = (clean, mixture, noise)

        downsamp_chan_envs = np.zeros((len(signals),
                                       np.ceil(N / self.downsamp_factor)
                                       .astype('int')))
        if (downsamp_chan_envs.shape[-1] % 2) == 0:
            len_offset = 1
        else:
            len_offset = 0
        chan_mod_envs = np.zeros((len(signals),
                                  len(self.modf),
                                  downsamp_chan_envs.shape[-1] - len_offset))
        time_av_mr_snr_env_matrix = np.zeros((N_cf, N_modf))
        lt_exc_ptns = np.zeros((len(signals), N_cf, N_modf))
        mr_snr_env_matrix = []
        mr_exc_ptns = []

        # find bands above threshold
        _, filtered_rms_mix = filterbank.noctave_filtering(mixture, self.cf,
                                                        self.fs, width=3)
        bands_above_thres_idx = self._bands_above_thres(filtered_rms_mix)

        for idx_band in bands_above_thres_idx:
            channel_envs = \
                    [self._peripheral_filtering(signal, self.cf[idx_band])
                     for signal in signals]

            for i_sig, channel_env in enumerate(channel_envs):
                # Extract envelope
                tmp_env = general.hilbert_envelope(channel_env).squeeze()

                # Low-pass filtering
                tmp_env = auditory.lowpass_env_filtering(tmp_env, 150.0,
                                                         N=1, fs=self.fs)
                # Downsample the envelope for faster processing
                downsamp_chan_envs[i_sig] = tmp_env[::self.downsamp_factor]

                # Sub-band modulation filtering
                lt_exc_ptns[i_sig, idx_band], chan_mod_envs[i_sig] = \
                    filterbank.mod_filterbank(downsamp_chan_envs[i_sig],
                                              fs_new,
                                              self.modf)

            chan_mr_exc_ptns = []
            for chan_env, mod_envs in izip(downsamp_chan_envs,
                                           chan_mod_envs):
                chan_mr_exc_ptns.append(self._mr_env_powers(chan_env, mod_envs))
            mr_exc_ptns.append(chan_mr_exc_ptns)

            time_av_mr_snr_env_matrix[idx_band], _, chan_mr_snr_env_matrix \
                = self._mr_snr_env(*chan_mr_exc_ptns[-2:])  # Select only the env
            # powers from the mixture and the noise, even if we calculated the
            # envelope powers for the clean speech.
            mr_snr_env_matrix.append(chan_mr_snr_env_matrix)

        lt_snr_env_matrix = super(MrSepsm, self)._snr_env(*lt_exc_ptns[-2:])
        lt_snr_env = self._optimal_combination(lt_snr_env_matrix,
                                             bands_above_thres_idx)

        snr_env = self._optimal_combination(time_av_mr_snr_env_matrix,
                                            bands_above_thres_idx)

        res = {
            'snr_env': snr_env,
            'snr_env_matrix': time_av_mr_snr_env_matrix,
            # .snr_env_matrix': snr_env_matrix

            # Output of what is essentially the sEPSM.
            'lt_snr_env': lt_snr_env,
            'lt_snr_env_matrix': lt_snr_env_matrix,
            'lt_exc_ptns': lt_exc_ptns,

            # .mr_snr_env': mr_snr_env
            'mr_snr_env_matrix': mr_snr_env_matrix,
            'mr_exc_ptns': mr_exc_ptns,

            'bands_above_thres_idx': bands_above_thres_idx
        }
        return res

    @staticmethod
    def _plot_mr_matrix(mat, x=None, y=None, fig=None, subplot_pos=111):
        """

        :param mat:
        :param x:
        :param y:
        :param ax:
        :return:
        """

        n_y, n_x = mat.shape
        if y is None:
            y = np.arange(n_y)

        max_mat = mat.max()
        bmap = brewer2mpl.get_map('PuBu','Sequential', 9).mpl_colormap

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

        for ax, p, f in izip(grid, mat[::-1], y[::-1]):
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
                         , vmax=None
                         , fig_subplt=None
                        ):
        """

        :param res: namedtuple, predictions from the model. Must have a
        `mr_snr_env_matrix` property.
        :param dur:
        :param db: bool, display dB values of the modulation power or SNRenv
        values. (Default: True.)
        :param vmax: float, maximum value of the colormap. If `None`,
        the data's maxium value is used. (Default: None)
        :return: self
        """

        mf = self.modf

        if vmax is None:
            vmax = ptns.max()
        if db:
            ptns = np.log10(ptns)
            cbar_label = 'SNRenv [dB]'
            vmax = np.log10(vmax)
        else:
            ptns = ptns
            cbar_label = 'SNRenv [lin]'

        n_mf, n_win = ptns.shape

        if dur is None:
            dur = n_win / self.modf[-1]

        if fig_subplt is None:
            fig = plt.figure()
            subplt = 111
        else:
            fig, subplt = fig_subplt

        bmap = brewer2mpl.get_map('PuBu','Sequential', 9).mpl_colormap
        xlabel = "Time [ms]"
        ylabel = "Modulation filter center frequency [Hz]"
        grid = ImageGrid(fig, subplt,
                         nrows_ncols=(n_mf, 1),
                         aspect=False,
                         # axes_pad=0.05,
                         # add_all=True,
                         share_all=False,
                         cbar_mode='single',
                         cbar_location='right',
                         cbar_size='0.5%',
                         cbar_pad=0.05)

        for ax, p, f in izip(grid, ptns[::-1], mf[::-1]):
            values = p.compressed()
            extent = (0, 1, 0, 1)
            im = ax.imshow(values[np.newaxis, :],
                      aspect='auto',
                      interpolation='none',
                      extent=extent,
                      vmax=vmax,
                      cmap=bmap)
            ax.grid(False)
            ax.set_yticks([0.5])
            ax.set_yticklabels([f])

        cbar = grid.cbar_axes[0].colorbar(im)
        cbar.ax.set_ylabel(cbar_label)
        grid[-1].set_xlabel(xlabel)
        grid[-1].set_xticks(np.arange(0, dur, 0.2))
        n_ticks = len(grid[-1].get_xticks())

        fig.text(0.05, 0.5, ylabel, va='center', rotation=90, size=11)


