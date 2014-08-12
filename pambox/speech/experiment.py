# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
from datetime import datetime
from itertools import product, islice

import collections
import logging
import os
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..utils import make_same_length, setdbspl, int2srt


log = logging.getLogger(__name__)


class Experiment(object):
    """
    Performs a speech intelligibility experiment.


    The masker is truncated to the length of the target, or is padded with
    zeros if it is shorter than the target.

    Parameters
    ----------
    models : single model, or list
        List of intelligibility models.
    materials : object
        object that implements a `next` interface that returns the next
        pair of target and maskers.
    snrs : array_like
        List of SNRs.
    name : str,
        name of experiment, will be appended to the date when writing to file.
    write : bool, optional
        Write the result to file, as CSV, the default is True.
    output_path : string, optional.
        Path where the results will be written if `write` is True. The
        default is './Data'.
    timestamp_format : str, optional
        Datetime timestamp format for the CSV file name. The default is of
        the form YYYYMMDD-HHMMSS.

    """

    def __init__(self, models, materials, snrs, fixed_level=65,
                 fixed_target=True, name=None,
                 write=True,
                 output_path='./Data/',
                 timestamp_format="%Y%m%d-%H%M%S"
    ):
        """@todo: to be defined1. """
        if isinstance(models, collections.Iterable):
            self.models = models
        else:
            self.models = [models]
        self.snrs = snrs
        self.timestamp_format = timestamp_format
        self.materials = materials
        self.write = write
        self.name = name
        self.fixed_level = fixed_level
        self.fixed_target = fixed_target
        self.output_path = output_path


    def preprocessing(self, target, masker):
        """
        Applies preprocessing to the target and masker before setting the
        levels. In this case, the masker is padded with zeros if it is longer
        than the target, or it is truncated to be the same length as the target.

        :param target:
        :param masker:
        :return:
        """
        if target.shape[-1] != masker.shape[-1]:
            target, masker = make_same_length(target, masker,
                                              extend_first=False)
        return target, masker

    def adjust_levels(self, target, masker, snr):
        """
        Adjusts level of target and maskers.

        Uses the `self.fixed_level` as the reference level for the target and
        masker. If `self.fixed_target` is True, the masker level is varied to
        set the required SNR, otherwise the target level is changed.

        :param target: ndarray
            Target signal.
        :param masker: ndarray
            Masker signal.
        :param snr: float
            SNR at which to set the target and masker.
        :return: tuple
            Level ajusted `target` and `masker`.
        """

        target_level = self.fixed_level
        masker_level = self.fixed_level
        if self.fixed_target:
            masker_level -= snr
        else:
            target_level += snr
        target = setdbspl(target, target_level, offset=0.0)
        masker = setdbspl(masker, masker_level, offset=0.0)
        return target, masker

    def append_results(self, df, res, model, snr):
        """
        Appends results to a DataFrame

        :param df: dataframe
            DataFrame where the new results will be appended.
        :param res: dict
            Output dictionary from an intelligibility model.
        :param model: object
            Intelligibility model. Will use it's `name` attribute,
            if available, to add the source model to the DataFrame. Otherwise,
            the `__class__.__name__` attribute will be used.
        :param snr: float
            SNR at which the simulation was performed.
        :return: dataframe
            DataFrame with new entry appended.
        """
        try:
            model_name = model.name
        except AttributeError:
            model_name = model.__class__.__name__
        d = {
            'SNR': snr
            , 'Model': model_name
        }
        for name, value in res['preds'].iteritems():
            d['Output'] = name
            d['Value'] = value
            df = df.append(d, ignore_index=True)

        return df

    def run(self, n=None):
        """ Run the experiment.

        Parameters
        ----------
        n : int
            number of conditions.

        Returns
        -------
        df : Pandas data frame with the experimental results.


        """

        # Initialize the dataframe in which the results are saved.
        df = pd.DataFrame()

        for ii, (model, snr, pair) in enumerate(product(self.models, self.snrs,
                                        islice(self.materials, n))):
            target, masker = pair

            target, masker = self.preprocessing(target, masker)

            target, masker = self.adjust_levels(target, masker, snr)

            log.info("Simulation # {}\t SNR: {}", ii, snr)
            res = self.prediction(model, target, masker)

            df = self.append_results(df, res, model, snr)

        if self.write:
            self.write_results(df)
        return df


    def write_results(self, df):
        """

        Writes results to CSV file.

        Parameters
        ----------
        df : dataframe

        Returns
        -------
        filepath : str
            Path to the CSV file.

        Raises
        ------
        IOError : Raise if the path where to write the CSV file is not
        accesssible. Additionally, the function tries to save the CSV file to
        the current directory, in order not to loose the simulation data.

        """
        timestamp = datetime.now()
        date = timestamp.strftime(self.timestamp_format)
        if self.name:
            name = "-{}".format(self.name)
        else:
            name = ''
        filename = "{date}{name}.csv".format(date=date, name=name)

        output_file = os.path.join(self.output_path, filename)
        try:
            df.to_csv(output_file)
            log.info('Saved CSV file to location: {}'.format(output_file))
        except IOError, e:
            try:
                alternate_path = os.path.join(os.getcwd(), filename)
                err_msg = 'Could not write CSV file to path: {}, tried to ' \
                          'save to ' \
                          '{} in order not to loose data.'.format(
                    output_file, alternate_path)
                log.error(err_msg)
                raise
            finally:
                try:
                    df.to_csv(alternate_path)
                except:
                    pass
        else:
            return output_file


    @staticmethod
    def prediction(model, target, masker):
        """
        Predicts intelligibility for a target and masker pair. The target and
        masker are simply added together to create the mixture.

        Parameters
        ----------
        model :
        target :
        masker :

        Returns
        -------
        :return:
        """
        return model.predict(target, target + masker, masker)


    def plot_results(self, df):
        df = df.ix[self.df['Target distance'] == 0.5, :]
        for key, grp in df.groupby(['Target distance', 'Masker distance']):
            plt.plot(grp['SNR'].unique(),
                     grp.groupby('SNR')['Intelligibility_L'].mean(),
                     label=str(key) + '_L')
            plt.plot(grp['SNR'].unique(),
                     grp.groupby('SNR')['Intelligibility_R'].mean(),
                     label=str(key) + '_R')
        srt_l = int2srt(grp['SNR'].unique(),
                        grp.groupby('SNR')['Intelligibility_L'].mean().values)
        srt_r = int2srt(grp['SNR'].unique(),
                        grp.groupby('SNR')['Intelligibility_R'].mean().values)
        try:
            print('Positions:%s\tSRTs L:%.1f\t R:%.1f' %
                  (key, srt_l[0], srt_r[0]))
        except:
            pass
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                   borderaxespad=0.)
        plt.xlabel('SNR (dB)')
        plt.ylabel('% Intelligibility')

    def snr_to_pc(self, df, col, fc, out_name='Intelligibility'):
        """
        Converts the data in a given column to percent correct.

        :param col: string, name of the column to convert
        :param fc: function, takes a float as input and returns a float.
        :param out_name: str, name of the output column (default:
        Intelligibility')
        """
        df[out_name] = df[col].map(fc)
        return df

    def get_srt(self, df):
        """Convert SRTs to DeltaSRTs.

        :return: tuple, srts and DeltaSRTs
        """

        scores = []
        df_tn = df[df['Target distance'] == 0.5]
        for ii, (key, grp) in enumerate(df_tn.groupby(['Target distance',
                                                       'Masker distance'])):
            scores.append(grp.groupby('SNR')['Intelligibility_L'].mean())

        srts = np.zeros(4)
        for ii, score in enumerate(scores):
            srts[ii] = int2srt(score.index, score)

        print(srts)
        dsrts = srts[1] - srts
        print(dsrts)
        return srts, dsrts


def srt_dict_to_dataframe(d):
    df_srts = pd.DataFrame()
    for k, v in d.iteritems():
        model, material, tdist, mdist = k.split('_')

        try:
            srt = v[0]
        except TypeError:
            srt = np.nan

        df_srts = df_srts.append({'model': model,
                                  'tidst': tdist,
                                  'mdist': mdist,
                                  'srt': srt},
                                 ignore_index=True)
    df_srts = df_srts.convert_objects(convert_numeric=True, )
    return df_srts.sort(['model', 'mdist'])


def plot_srt_dataframe(df):
    for key, grp in df.groupby('model'):
        plt.plot(grp['mdist'], grp['srt'], label=key)
    plt.legend(loc='best')
    plt.xlabel('Masker distance')
    plt.ylabel('SRT (dB)')
