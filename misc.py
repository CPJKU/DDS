import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import reduce
from operator import getitem


def inceptdict():
    '''Returns dict that creates new instance of itself on non-existent key access.'''
    return defaultdict(inceptdict)


def set_nested_item(nested_dict, key_sequence, value):
    '''Set item in nested dictionary.

    Parameters
    ----------
    nested_dict : dict
        Nested dictionary (dictionary of dictionaries) with arbitrarily deep structure.
    key_sequence : list
        List of keys to access the "leaf" dictionary item of which value is to be set.
    value : type
        Value to set to the particular item.

    Returns
    -------
    dict
        The nested dictionary in which value was set.
    '''
    reduce(getitem, key_sequence[:-1], nested_dict)[key_sequence[-1]] = value
    return nested_dict


def get_nested_item(nested_dict, key_sequence):
    '''Get item in nested dictionary.

    Parameters
    ----------
    nested_dict : dict
        Nested dictionary (dictionary of dictionaries) with arbitrarily deep structure.
    key_sequence : list
        List of keys to access the "leaf" dictionary item of which value is to be retrieved.

    Returns
    -------
    type
        The retrieved item.
    '''
    return reduce(getitem, key_sequence, nested_dict)


def find_files(directory, pattern, path=True):
    '''Recursively finds all files matching the pattern.

    Parameters
    ----------
    directory : str
        The directory to search in.
    pattern : str
        File name pattern to match.
    path :
        Whether strings should include paths (the default is True).

    Returns
    -------
        List of strings with the [paths/] names of matched files.
    '''

    files = []
    for root, dirnames, filenames in os.walk(
        directory, followlinks=True):
        for filename in fnmatch.filter(filenames, pattern):
            if path:
                files.append(os.path.join(root, filename))
            else:
                files.append(filename)
    return files


# log/delog magnitude spectrogram
spec_log = lambda S: np.log(1 + np.abs(S))
spec_delog = lambda S: np.exp(S) - 1

# norm/denorm magnitude spectrogram
def spec_norm(S, win_length, log=True, strength=100):
    '''Normalizes magnitude spectrogram to range [0; 1], given
    that hanning window of size `win_length` was used to compute
    the STFT. Optionally logarithmizes features to amplify the
    low-magnitude components.

    Parameters
    ----------
    S : numpy.ndarray
        Spectral magnitudes to normalize.
    win_length : int
        Length of the (hanning) window used to compute the STFT.
    log : bool
        Whether to apply logarithmization (the default is True).
    strength : int
        Data scale on which log is applied (the default is 100).
        Call `plot_log_strengths()` for illustration of how different
        values influence the projection on the interval [0; 1].

    Returns
    -------
    numpy.ndarray
        Normalized (logarithmized) magnitude spectrogram.
    '''

    # normalize to [0; 1]
    S = 2 * S / np.sum(np.hanning(win_length))
    # logarithmize
    if log:
        S = np.log1p(S * strength) / np.log1p(strength)
    return S

def spec_denorm(S, win_length, log=True, strength=100):
    '''Reverses the normalization of magnitude spectrogram.
    Optinal logarithmization is reversed as well.

    For correct de-normalization, use same values as in `spec_norm`.

    Parameters
    ----------
    S : numpy.ndarray
        Spectral magnitudes to de-normalize.
    win_length : int
        Length of the (hanning) window used to compute the STFT.
    log : bool
        Whether to reverse logarithmization (the default is True).
    strength : int
        Data scale on which log is applied (the default is 100).

    Returns
    -------
    numpy.ndarray
        De-Normalized (de-logarithmized) magnitude spectrogram.
    '''

    # reverse the logarithmization
    if log:
        S = (np.exp(S * np.log1p(strength)) - 1) / strength
    # reverse the normalization
    S = (1/2) * S * np.sum(np.hanning(win_length))
    return S

def plot_log_strengths(strengths=[1e1, 1e2, 1e3, 1e4, 1e5]):
    '''Illustrates the influence of `strength` parameter in
    `spec_norm` function on a set of different `strength` values.

    Parameters
    ----------
    strengths : list
        Values of `strength` for which to illustrate the
        logarithmization (the default is [1e1, 1e2, 1e3, 1e4, 1e5]).

    Returns
    -------
    None
    '''

    data = np.arange(100) / 100

    plt.plot(data, label='data')

    for s in strengths:
        log_rescaled = np.log1p(data*s) / np.log1p(s)
        plt.plot(log_rescaled, label=f'log strength = {int(s)}')

    plt.legend()
    plt.title('Influence of `strength` parameter in `spec_norm`.')
