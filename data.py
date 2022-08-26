import os
import fnmatch
import numpy as np
import torch
from torch.utils.data import Dataset

import librosa
import pretty_midi as pm
pm.pretty_midi.MAX_TICK = 1e10

from misc import find_files, spec_norm, inceptdict, set_nested_item, get_nested_item


def extract_timbre_frames(note_number, datadir,
                          sr=22050, n_fft=512, hop_length=512//4,
                          log_str=10000, keep_bins=0, keep_frames=0):
    '''Given `datadir` location containing single midi file descriptive
    of one or more wav files (renderings of the midi file by different instruments),
    computes magnitude spectrograms of all wav files and extracts all time frames
    where note of pitch `note_number` is active.

    Parameters
    ----------
    note_number : int
        MIDI number of the note to extract frames for.
    datadir : str
        Directory containing midi and audio rendition(s) of specified note.
    sr : int
        Sampling rate to use with the audio (the default is 22050).
    n_fft : int
        Size of the hanning window used with STFT in samples (the default is 512).
    hop_length : int
        Size of the shift of the STFT window in samples (the default is 512//4).
    log_str : int
        Data scale on which log is applied (the default is 10000).
        See `misc.spec_norm()` for details.
    keep_bins : int
        Amount of frequency bins to keep (the default is 0 and stands for keeping
        all the frequency bins).
    keep_frames : int
        The (non-zero) amount of time frames to keep from the start of
        each note (the default is 0 and stands for keeping all the time frames).

    Returns
    -------
    dict
        Nested dictionary of dictionaries grouping extracted spectral frames
        as `numpy.ndarray`s together by common instruments and velocities.
    '''

    file_mid = find_files(datadir, f'*.mid')[0]
    midi = pm.PrettyMIDI(file_mid)

    velocities = []; starts = []; ends = [];
    for instrument in midi.instruments:
        for note in instrument.notes:
            if note.pitch == note_number:
                velocities.append(note.velocity)
                starts.append(note.start)
                ends.append(note.end)

    assert len(velocities) > 0, f'No instance of note {pm.note_number_to_name(note_number)} found in {file_mid}'

    files_wav = find_files(datadir, f'*.wav')

    frames = {}
    for file in files_wav:

        instrument = file.split('/')[-1].split('.')[0]

        y, _ = librosa.load(file, sr=sr)

        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        S, _ = librosa.core.magphase(S[1:, :]) # discard lowest (DC) bin (1+n_fft//2 --> n_fft//2)
        S = spec_norm(S, n_fft, log=log_str>0, strength=log_str)

        if keep_bins:
            S = S[:keep_bins, :] # low-pass (discard higher frequency bins)

        sec2frame = S.shape[1] / (len(y) / sr)

        frames[instrument] = {}

        for velocity, start, end in zip(velocities, starts, ends):
            frame_from = int(np.round(start * sec2frame))
            frame_to = int(np.round(end * sec2frame))
            if keep_frames:
                frame_to = min(frame_to, frame_from + keep_frames)

            frames[instrument][velocity] = S[:, frame_from:frame_to]

    return frames


class SingleNoteTimbreFramesFactory(object):
    '''Creates a torch.utils.data.Dataset factory that is initialized
    with spectral frames of a single note capturing timbral variation
    in extracting the note performed by various instruments with various
    velocity values from the files contained in the provided directory.

    After first instantiation, caches the set of extracted frames
    for future re-use in `./data/{config_string}/{note_string}.npy`.

    Call member class method `.make_dataset()` to generate a
    torch.utils.data.Dataset object to be used with a model.

    Parameters
    ----------
    note_name : str
        Name of note pitch to extract as per the `pretty_midi` reference.
        https://craffel.github.io/pretty-midi/#pretty_midi.note_name_to_number
    datadir : str
        Path to directory where midi labels and wav files are contained.
    sr : int
        Sampling rate to use with the audio (the default is 22050).
    n_fft : int
        Size of the hanning window used with STFT in samples (the default is 512).
    hop_length : int
        Size of the shift of the STFT window in samples (the default is 512//4).
    log_str : int
        Data scale on which log is applied (the default is 10000).
        See `misc.spec_norm()` for details.
    keep_bins : int
        Amount of frequency bins to keep (the default is 0 and stands for keeping
        all the frequency bins).
    keep_frames : int
        The (non-zero) amount of time frames to keep from the start of
        each note (the default is 0 and stands for keeping all the time frames).

    Attributes
    ----------
    instruments : dict
        Dictionary of filenames corresponding to different instruments
        as paired to indices to use for instrument filtering.
    velocities : set
        Set of velocity values contained in the dataset. To use
        velocity filtering.
    frames_dict : dict
        Nested dictionary of dictionaries grouping extracted spectral frames
        as `numpy.ndarray`s together by common instruments and velocities,
        as produced by `data.extract_timbre_frames()`.
    make_dataset : method
        Memer class method that creates a Dataset as a subset of all samples
        based on timbral (velocity / instrument) and other filtering criteria.
    '''

    def __init__(self, note_name, datadir,
                 sr=22050, n_fft=512, hop_length=512//4, log_str=10000,
                 keep_bins=0, keep_frames=0):
        source_dir = list(filter(None, datadir.split('/')))[-1]
        note_number = pm.note_name_to_number(note_name)
        note_string = f'KS_{source_dir}_{note_number:03}_{note_name}'
        config_string = f'{sr}_{n_fft}_{hop_length}_{log_str}_{keep_bins}_{keep_frames}'
        config_savepath = f'data/{config_string}/{note_string}.npy'

        self.config_string = config_string

        if not os.path.isfile(config_savepath):
            frames = extract_timbre_frames(
                note_number, datadir, sr, n_fft, hop_length, log_str, keep_bins, keep_frames)
            os.makedirs(f'data/{config_string}', exist_ok=True)
            np.save(config_savepath, frames)
        else:
            frames = np.load(config_savepath, allow_pickle=True)[np.newaxis][0]

        frames = dict(sorted(frames.items()))
        self.instruments = dict(enumerate(frames.keys()))
        self.velocities = set(list(frames.values())[0].keys())
        self.frames_dict = frames

    def _filter_frames(self, inst_min, inst_max, vel_min, vel_max, inst_set, vel_set,
                       keep_bins, keep_frames):
        MAX_BINS = 100000
        MAX_FRAMES = 10000000
        frames_list = []

        for instrument, velocities in enumerate(self.frames_dict.values()):
            if instrument >= inst_min and instrument <= inst_max and (
                instrument in inst_set if bool(inst_set) else True):
                for velocity, frames in velocities.items():
                    if velocity >= vel_min and velocity <= vel_max and (
                        velocity in vel_set if bool(vel_set) else True):
                        frames_list += [frames[:keep_bins if keep_bins else MAX_BINS,
                                               :keep_frames if keep_frames else MAX_FRAMES]]

        return np.concatenate(frames_list, axis=1)

    def plot_filtered_samples(self, note_name, sr, hop_length, figsize=(16, 12),
                    inst_min=0, inst_max=42, vel_min=1, vel_max=127,
                    inst_set=set(), vel_set=set(), keep_bins=0, keep_frames=0):
        # TODO docstring
        from librosa.display import specshow
        import matplotlib.pyplot as plt

        MAX_BINS = 100000
        MAX_FRAMES = 10000000

        note_number = pm.note_name_to_number(note_name)
        savedir = f'vis/{self.config_string}'
        os.makedirs(savedir, exist_ok=True)

        for instrument, velocities in enumerate(self.frames_dict.values()):
            if instrument >= inst_min and instrument <= inst_max and (
                instrument in inst_set if bool(inst_set) else True):
                for velocity, frames in velocities.items():
                    if velocity >= vel_min and velocity <= vel_max and (
                        velocity in vel_set if bool(vel_set) else True):
                        # extract
                        data = frames[:keep_bins if keep_bins else MAX_BINS,
                                      :keep_frames if keep_frames else MAX_FRAMES]
                        # plot
                        fig = plt.figure(figsize=figsize)
                        specshow(data, sr=sr, hop_length=hop_length, cmap='Greys')
                        plt.title(f'{note_name} | inst {instrument:03} | vel {velocity:03}')
                        savepath = f'{savedir}/{note_number:03}_{instrument:03}_{velocity:03}_{note_name}.png'
                        plt.savefig(savepath, bbox_inches='tight')
                        plt.close(fig)

    def make_dataset(self, inst_min=0, inst_max=42,
                    vel_min=1, vel_max=127,
                    inst_set=set(), vel_set=set(),
                    keep_bins=0, keep_frames=0, min_energy=0,
                    shuffle=False, split='full', transform=None):
        '''Extracts frames of subset of velocities and instruments as specified
        by filter arguments (either explicit sub-set or just interval) and applies
        optional energy filtering, shuffling and subset selection (as configured
        by arguments passed at dataset initialization). Stores the resulting
        dataset as torch tensor in member attribute `x` for dataset to use.

        If both interval and set are provided for a particular axis, both constraints
        are applied via logical AND for sample filtering.

        Parameters
        ----------
        inst_min : int
            Lower interval boundary for instrument index (the default is 0).
        inst_max : int
            Upper interval boundary for instrument index (the default is 42).
        vel_min : int
            Lower interval boundary for velocity (the default is 1).
        vel_max : int
            Upper interval boundary for velocity (the default is 127).
        inst_set : set
            Explicit discriminative set of instruments to include (the default is set()).
        vel_set : set
            Explicit discriminative set of velocities to include (the default is set()).
        keep_bins : int
            Amount of frequency bins to keep (the default is 0 and stands for keeping
            all the frequency bins).
        keep_frames : int
            The (non-zero) amount of time frames to keep from the start of
            each note (the default is 0 and stands for keeping all the time frames).
        min_energy : int
            Threshold for discarding time frames with low energy (the default is 0).
            All magnitude spectral frames with total energy (sum of magnitudes
            over frequency bins) below this threshold will be excluded from the dataset,
            except for 0 which will bypass this filter.
        shuffle : bool
            Whether to (deterministically) shuffle the resulting set of frames
            before the final split is created (the default is False).
        split : str
            Which portion of the dataset to use (the default is 'full').
            Options:
             - 'full' gives all resulting frames.
             - 'train' gives first 80% of the frames (after the shuffling).
             - 'valid' gives last 20% of the frames (after the shuffling).
        transform : obj
            A transform object following the standard interface of
            `torchvision.transforms.*` transform objects (the default is None).

        Returns
        -------
        SingleNoteTimbreFrames
            Object that is a subclass of torch.utils.data.Dataset.
        '''

        frames = self._filter_frames(
            inst_min, inst_max, vel_min, vel_max, inst_set, vel_set, keep_bins, keep_frames)

        frames = frames.T # [freq x samples] -> [samples x freq]

        if min_energy:
            energy = np.sum(frames, axis=1)
            frames = frames[energy > min_energy]

        subset = np.arange(len(frames))
        if shuffle:
            np.random.seed(0) # reproducible split via fixed seed
            np.random.shuffle(subset)
            np.random.seed() # re-randomize numpy.random ops from here

        subset = {'train': subset[:int(len(frames)*0.8)],
                  'valid': subset[int(len(frames)*0.8):],
                  'full': subset}
        frames = frames[subset[split]]

        return SingleNoteTimbreFrames(frames, transform)


class SingleNoteTimbreFrames(Dataset):
    def __init__(self, frames, transform):
        self.x = torch.tensor(frames).float()
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
        return x, list() # return dummy labels list


class MultiNoteTimbreFramesFactory(object):
    '''Wrapper of `SingleNoteTimbreFramesFactory` that combines multiple SingleNote
    factories to generate a factory for a whole range of notes.
    Extends SingleNoteTimbreFrames to multiple sources similarly to how
    `MAPS_ISOL_NoteFrames` is extended by `MAPS_ISOL_NoteFrames_MultiSource`.
    '''

    def __init__(self, note_range, datadir,
                 sr=22050, n_fft=512, hop_length=512//4, log_str=10000,
                 keep_bins=0, keep_frames=0):

        note_from, note_to = pm.note_name_to_number(note_range[0]), pm.note_name_to_number(note_range[1])

        from tqdm import tqdm
        factories = []

        for note_number in tqdm(range(note_from, note_to+1), desc='Loading SingleNote datasets'):
            note_name = pm.note_number_to_name(note_number)
            dataset_factory = SingleNoteTimbreFramesFactory(
                note_name, datadir, sr, n_fft, hop_length, log_str, keep_bins, keep_frames)
            factories.append(dataset_factory)

        self.single_note_ds_factories = factories

    def make_dataset(self, transform=None, **kwargs):

        x = list()
        y = list()

        for i, dataset_factory in enumerate(self.single_note_ds_factories):
            dataset = dataset_factory.make_dataset(transform=transform, **kwargs)
            x.append(dataset.x)
            y.append(torch.tensor([i] * len(dataset.x)))

        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)

        return MultiNoteTimbreFrames(x, y, transform)


class MultiNoteTimbreFrames(Dataset):
    def __init__(self, x, y, transform):
        self.x = x #torch.tensor(x).float()
        self.y = y #torch.tensor(y).long()
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


class JitterTransform:
    '''Adds noise of magnitude `amount` to samples assumed
    to be on interval [0; 1] and re-normalizes to stay on this interval.

    Parameters
    ----------
    amount : float
        Scale of the uniform noise to add (the default is 1/256).
    '''

    def __init__(self, amount=1/256):
        self.amount = amount

    def __call__(self, x):
        return (x + torch.rand_like(x) * self.amount) / (1 + self.amount)


class SnippetFramesFactory(object):
    def __init__(self, snippet, datadir,
                 sr=22050, n_fft=512, hop_length=512//4, log_str=10000,
                 keep_bins=0, keep_frames=0):
        config_string = f'{sr}_{n_fft}_{hop_length}_{log_str}_{keep_bins}_{keep_frames}'
        config_savepath = f'data/{config_string}/{snippet}.npy'

        if not os.path.isfile(config_savepath):
            frames = self.load_snippet(
                snippet, datadir, sr, n_fft, hop_length, log_str, keep_bins, keep_frames)
            os.makedirs(f'data/{config_string}', exist_ok=True)
            np.save(config_savepath, frames)
        else:
            frames = np.load(config_savepath, allow_pickle=True)[np.newaxis][0]

        file_mid = find_files(f'{datadir}/{snippet}', f'*.mid')[0]
        self.mid = pm.PrettyMIDI(file_mid)

        self.snippet_name = snippet
        self.config_string = config_string
        self.frames_dict = dict(sorted(frames.items()))
        self.instruments = list(frames.keys())

        self.sr = sr
        self.hop_length = hop_length

    def load_snippet(self, snippet, datadir,
                     sr=22050, n_fft=512, hop_length=512//4,
                     log_str=10000, keep_bins=0, keep_frames=0):
        files_wav = find_files(datadir, f'*.wav')

        frames = {}
        for file in files_wav:

            instrument = file.split('/')[-1].split('.')[0]

            y, _ = librosa.load(file, sr=sr)

            S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            S, _ = librosa.core.magphase(S[1:, :]) # discard lowest (DC) bin (1+n_fft//2 --> n_fft//2)
            S = spec_norm(S, n_fft, log=log_str>0, strength=log_str)

            if keep_bins:
                S = S[:keep_bins, :] # low-pass (discard higher frequency bins)

            if keep_frames:
                S = S[:, :keep_frames]

            frames[instrument] = S

        return frames

    def get_full_snippet(self, instrument, torchify=True):
        assert instrument in self.instruments
        snippet = self.frames_dict[instrument]
        return torch.tensor(snippet).float() if torchify else snippet

    def plot_full_snippet(self, instrument, figsize):
        from librosa.display import specshow
        import matplotlib.pyplot as plt

        snippet = self.frames_dict[instrument]
        fig = plt.figure(figsize=figsize)
        specshow(snippet, sr=self.sr, hop_length=self.hop_length, cmap='Greys')
        plt.title(f'snippet {self.snippet_name} | inst {instrument}')
        return fig

    def plot_full_pianoroll(self, figsize, inst=0, limit_range=False):
        from librosa.display import specshow
        import matplotlib.pyplot as plt

        pianoroll_full = self.mid.get_piano_roll(fs=float(self.sr/self.hop_length)) # int() quantization here displaces labels

        # 0-pad the labels on the right to account for release part of ADSR of final note(s)
        T_S = self.frames_dict[self.instruments[inst]].shape[1] # snippet's number of time frames
        T_P = pianoroll_full.shape[1] # piano-roll's number of time frames
        if T_S > T_P:
            pianoroll_full = np.pad(pianoroll_full, ((0, 0), (0, T_S - T_P)), constant_values=0)

        if limit_range:
            notes_present = np.unique(np.concatenate([[n.pitch for n in i.notes] for i in self.mid.instruments]))
            pianoroll = pianoroll_full[min(notes_present):max(notes_present)+1]
        else:
            pianoroll = pianoroll_full

        fig = plt.figure(figsize=figsize)
        ax = plt.gca(); ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False);
        specshow(pianoroll, x_axis='frames', y_axis='frames', cmap='Greys');
        return fig


class MAPS_ISOL_NoteFrames(Dataset):
    '''
    realistic=True and split=full yields the train+valid parts of the
    realistic split (s.t. the test split ENSTD* is not included).'''
    def __init__(self, note_name, mapsdir='data/MAPS/', sr=22050, n_fft=512, hop_length=512//4, log_str=10000,
                 keep_bins=0, keep_frames=0, min_energy=0,
                 split='train', realistic=False, transform=None, shuffle=True):
        if split == 'test':
            assert realistic, 'Test set is only defined for realistic split.'
        self.transform = transform

        note_number = pm.note_name_to_number(note_name)
        note_string = f'MAPS_{note_number:03}_{note_name}' + (f'_{split}' if realistic else '')

        config_string = f'{sr}_{n_fft}_{hop_length}_{log_str}_{keep_bins}_{keep_frames}'
        config_savepath = f'data/{config_string}/{note_string}.npy'

        if not os.path.isfile(config_savepath):
            frames = self.extract_frames(
                note_number, mapsdir, sr, n_fft, hop_length, log_str, keep_bins, keep_frames, split, realistic)
            os.makedirs(f'data/{config_string}', exist_ok=True)
            np.save(config_savepath, frames)
        else:
            frames = np.load(config_savepath)
        frames = frames.T

        if min_energy:
            energy = np.sum(frames, axis=1)
            frames = frames[energy > min_energy]

        subset = np.arange(len(frames))
        if shuffle:
            np.random.seed(0) # reproducible split via fixed seed
            np.random.shuffle(subset)
            np.random.seed() # re-randomize numpy.random ops from here
        subset = {'train': subset[:int(len(frames)*0.8)], 'valid': subset[int(len(frames)*0.8):], 'test': subset, 'full': subset}
        frames = frames[subset[split]]

        self.x = torch.tensor(frames).float()

    def extract_frames(self, note_number, mapsdir, sr=22050, n_fft=512, hop_length=512//4, log_str=10000,
                       keep_bins=0, keep_frames=0, split='train', realistic=False):
        files = find_files(mapsdir, f'*_M{note_number}_*.wav')
        files = [f[:-4] for f in files if not 'TR' in f]
        if realistic:
            filter_func = lambda f: 'ENSTD' in f if split=='test' else 'ENSTD' not in f
            files = list(filter(filter_func, files))

        frames = []

        for file in files:
            y, _ = librosa.load(file+'.wav', sr=sr)

            S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            S, _ = librosa.core.magphase(S[1:, :]) # discard lowest (DC) bin (1+n_fft//2 --> n_fft//2)
            S = spec_norm(S, n_fft, log=log_str>0, strength=log_str)

            if keep_bins:
                S = S[:keep_bins, :] # low-pass (discard higher frequency bins)

            sec2frame = S.shape[1] / (len(y) / sr)

            m = pm.PrettyMIDI(file+'.mid')

            starts = []; ends = [];
            for i in m.instruments:
                for n in i.notes:
                    if n.pitch == note_number:
                        frame_from = int(np.round(n.start * sec2frame))
                        frame_to = int(np.round(n.end * sec2frame))
                        if keep_frames:
                            frame_to = min(frame_to, frame_from + keep_frames)

                        frames.append(S[:, frame_from:frame_to])

        return np.concatenate(frames, axis=1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
        return x, list()


class MAPS_ISOL_NoteFrames_MultiSource(Dataset):
    '''
    realistic=True and split=full yields the train+valid parts of the
    realistic split (s.t. the test split ENSTD* is not included).
    '''
    def __init__(self, note_range, mapsdir='data/MAPS/', sr=22050, n_fft=512, hop_length=512//4, log_str=10000,
                 keep_bins=0, keep_frames=0, min_energy=0,
                 split='train', realistic=False, transform=None, shuffle=True):
        assert len(note_range) == 2
        note_from, note_to = pm.note_name_to_number(note_range[0]), pm.note_name_to_number(note_range[1])
        assert note_from < note_to
        self.note_range = note_range

        if split == 'test':
            assert realistic, 'Test set is only defined for realistic split.'
        self.transform = transform

        x = list()
        y = list()
        for i, note_number in enumerate(range(note_from, note_to+1)):
            note_name = pm.note_number_to_name(note_number)
            note_string = f'MAPS_{note_number:03}_{note_name}' + (f'_{split}' if realistic else '')

            config_string = f'{sr}_{n_fft}_{hop_length}_{log_str}_{keep_bins}_{keep_frames}'
            config_savepath = f'data/{config_string}/{note_string}.npy'

            if not os.path.isfile(config_savepath):
                frames = self.extract_frames(
                    note_number, mapsdir, sr, n_fft, hop_length, log_str, keep_bins, keep_frames, split, realistic)
                os.makedirs(f'data/{config_string}', exist_ok=True)
                np.save(config_savepath, frames)
            else:
                frames = np.load(config_savepath)
            frames = frames.T # [bins x samples] --> [samples x bins]

            if min_energy:
                energy = np.sum(frames, axis=1)
                frames = frames[energy > min_energy]

            subset = np.arange(len(frames))
            if shuffle:
                np.random.seed(0) # reproducible split via fixed seed
                np.random.shuffle(subset)
                np.random.seed() # re-randomize numpy.random ops from here
            subset = {'train': subset[:int(len(frames)*0.8)], 'valid': subset[int(len(frames)*0.8):], 'test': subset, 'full': subset}
            frames = frames[subset[split]]

            x.append(torch.tensor(frames).float())
            y.append(torch.tensor([i] * len(subset[split])))

        self.x = torch.cat(x, dim=0)
        self.y = torch.cat(y, dim=0)


    def extract_frames(self, note_number, mapsdir, sr=22050, n_fft=512, hop_length=512//4, log_str=10000,
                       keep_bins=0, keep_frames=0, split='train', realistic=False):
        files = find_files(mapsdir, f'*_M{note_number}_*.wav')
        files = [f[:-4] for f in files if not 'TR' in f]
        if realistic:
            filter_func = lambda f: 'ENSTD' in f if split=='test' else 'ENSTD' not in f
            files = list(filter(filter_func, files))

        frames = []

        for file in files:
            y, _ = librosa.load(file+'.wav', sr=sr)

            S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            S, _ = librosa.core.magphase(S[1:, :]) # discard lowest (DC) bin (1+n_fft//2 --> n_fft//2)
            S = spec_norm(S, n_fft, log=log_str>0, strength=log_str)

            if keep_bins:
                S = S[:keep_bins, :] # low-pass (discard higher frequency bins)

            sec2frame = S.shape[1] / (len(y) / sr)

            m = pm.PrettyMIDI(file+'.mid')

            starts = []; ends = [];
            for i in m.instruments:
                for n in i.notes:
                    if n.pitch == note_number:
                        frame_from = int(np.round(n.start * sec2frame))
                        frame_to = int(np.round(n.end * sec2frame))
                        if keep_frames:
                            frame_to = min(frame_to, frame_from + keep_frames)

                        frames.append(S[:, frame_from:frame_to])

        return np.concatenate(frames, axis=1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


class MAPS_ChordsFramesFactory(object):
    def __init__(self, subset, instr=8, mapsdir='data/MAPS/',
                 sr=22050, n_fft=512, hop_length=512//4, log_str=10000,
                 keep_bins=0, legacy=False):
        assert subset in ['RAND', 'UCHO']
        assert instr in range(9)

        self.instruments = ['StbgTGd2', 'AkPnBsdf', 'AkPnBcht', 'AkPnCGdD', 'AkPnStgb',
                            'SptkBGAm', 'SptkBGCl', 'ENSTDkAm', 'ENSTDkCl']
        self.instrument = self.instruments[instr]
        self.subset = subset
        self.legacy = legacy

        config_string = f'{sr}_{n_fft}_{hop_length}_{log_str}_{keep_bins}' + ('_legacy' if legacy else '')
        config_savepath = f'data/{config_string}/{subset}_{self.instrument}.npy'

        if not os.path.isfile(config_savepath):
            frames = self.load_frames(
                f'{mapsdir}/{self.instrument}/{self.subset}',
                sr, n_fft, hop_length, log_str, keep_bins)
            os.makedirs(f'data/{config_string}', exist_ok=True)
            np.save(config_savepath, frames)
        else:
            frames = np.load(config_savepath, allow_pickle=True)[np.newaxis][0]

        self.config_string = config_string
        self.frames_dict = dict(sorted(frames.items()))

        self.sr = sr
        self.hop_length = hop_length

    def load_frames(self, directory, sr=22050, n_fft=512, hop_length=512//4, log_str=10000, keep_bins=0):
        frames_dict = inceptdict()

        for root, dirnames, filenames in os.walk(directory, followlinks=True):
            key_seq = list(filter(None, root.replace(directory, '').split('/')))

            if len(filenames) > 0:
                frames = []
                rolls = []

            for filename in sorted(fnmatch.filter(filenames, '*.wav')):
                file_wav = os.path.join(root, filename)
                file_mid = file_wav.replace('.wav', '.mid')

                midi = pm.PrettyMIDI(file_mid)
                R = midi.get_piano_roll(fs=float(sr/hop_length))
                notes_start = min(np.concatenate([[n.start for n in i.notes] for i in midi.instruments]))
                notes_end = max(np.concatenate([[n.end for n in i.notes] for i in midi.instruments]))

                y, _ = librosa.load(file_wav, sr=sr)

                S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
                if self.legacy:
                    S, _ = librosa.core.magphase(S[:-1, :]) # discard highest bin (1+n_fft//2 --> n_fft//2)
                else:
                    S, _ = librosa.core.magphase(S[1:, :]) # discard lowest (DC) bin (1+n_fft//2 --> n_fft//2)
                S = spec_norm(S, n_fft, log=log_str>0, strength=log_str)

                if keep_bins:
                    S = S[:keep_bins, :] # low-pass (discard higher frequency bins)

                sec2frame = S.shape[1] / (len(y) / sr)

                frame_from = int(np.round(notes_start * sec2frame))
                frame_to = int(np.round(notes_end * sec2frame))

                # handle potential mis-shape-ing of (x,y) pairs caused by `np.round()` above
                if frame_to > R.shape[1]:
                    assert frame_to - R.shape[1] == 1, 'Rounding can at most produce an error of 1 frame'
                    R_e = R[:, -1][:, np.newaxis] # extract last column (time frame)
                    R = np.concatenate([R, R_e], axis=1) # duplicate this column at the end

                frames.append(S[:, frame_from:frame_to])
                rolls.append(R[:, frame_from:frame_to])

            if len(filenames) > 0:
                set_nested_item(frames_dict, key_seq + ['x'], np.concatenate(frames, axis=1))
                set_nested_item(frames_dict, key_seq + ['y'], np.concatenate(rolls, axis=1))

        return frames_dict

    def get_frames(self, key_seq, torchify=True):
        frames = get_nested_item(self.frames_dict, key_seq)['x']
        return torch.tensor(frames).float() if torchify else frames

    def get_pianoroll(self, key_seq):
        pianoroll = get_nested_item(self.frames_dict, key_seq)['y']
        return pianoroll

    def plot_frames(self, key_seq, figsize):
        from librosa.display import specshow
        import matplotlib.pyplot as plt

        frames = get_nested_item(self.frames_dict, key_seq)['x']
        fig = plt.figure(figsize=figsize)
        specshow(frames, sr=self.sr, hop_length=self.hop_length, cmap='Greys')
        plt.title(f'{self.subset} | {self.instrument} | ' + ' | '.join(key_seq))
        return fig

    def plot_pianoroll(self, key_seq, figsize, limit_range=False, limits=None):
        from librosa.display import specshow
        import matplotlib.pyplot as plt

        pianoroll_full = get_nested_item(self.frames_dict, key_seq)['y']

        if limit_range:
            if limits is not None and len(limits) == 2:
                min_notes, max_notes = limits
            else:
                noteset = np.argwhere(pianoroll_full.sum(axis=1))
                min_notes = noteset.min()
                max_notes = noteset.max()
            pianoroll = pianoroll_full[min_notes:max_notes+1]
        else:
            pianoroll = pianoroll_full

        fig = plt.figure(figsize=figsize)
        ax = plt.gca(); ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False);
        specshow(pianoroll, x_axis='frames', y_axis='frames', cmap='Greys');
        plt.title(f'{self.subset} | {self.instrument} | ' + ' | '.join(key_seq))
        return fig


class MAPS_MUS_PieceFramesFactory(object):
    def __init__(self, instr=8, mapsdir='data/MAPS/',
                 sr=22050, n_fft=512, hop_length=512//4, log_str=10000,
                 keep_bins=0, note_range_limit=None):
        assert instr in range(9)

        self.instruments = ['StbgTGd2', 'AkPnBsdf', 'AkPnBcht', 'AkPnCGdD', 'AkPnStgb',
                            'SptkBGAm', 'SptkBGCl', 'ENSTDkAm', 'ENSTDkCl']
        self.instrument = self.instruments[instr]

        config_string = f'{sr}_{n_fft}_{hop_length}_{log_str}_{keep_bins}'
        config_savepath = f'data/{config_string}/MUS_{self.instrument}.npy'

        if not os.path.isfile(config_savepath):
            frames = self.load_frames(
                f'{mapsdir}/{self.instrument}/MUS',
                sr, n_fft, hop_length, log_str, keep_bins)
            os.makedirs(f'data/{config_string}', exist_ok=True)
            np.save(config_savepath, frames)
        else:
            frames = np.load(config_savepath, allow_pickle=True)[np.newaxis][0]

        self.config_string = config_string
        self.frames_dict = dict(sorted(frames.items()))

        self.sr = sr
        self.hop_length = hop_length
        # self.note_range_limit = note_range_limit

    def load_frames(self, directory, sr=22050, n_fft=512, hop_length=512//4, log_str=10000, keep_bins=0):
        frames_dict = {}

        files = [f[:-4] for f in find_files(directory, '*.wav')]

        for file in files:
            piece_name = file.split('/')[-1]
            frames_dict[piece_name] = {}

            midi = pm.PrettyMIDI(f'{file}.mid')
            R = midi.get_piano_roll(fs=float(sr/hop_length))
            # truncation of fs to int() here would introduce systematic and accumulating temporal displacement of labels

            y, _ = librosa.load(f'{file}.wav', sr=sr)

            S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            S, _ = librosa.core.magphase(S[1:, :]) # discard lowest (DC) bin (1+n_fft//2 --> n_fft//2)
            S = spec_norm(S, n_fft, log=log_str>0, strength=log_str)

            if keep_bins:
                S = S[:keep_bins, :] # low-pass (discard higher frequency bins)

            if S.shape[1] > R.shape[1]:
                print(f'Shape mismatch: S={S.shape[1]} R={R.shape[1]}. Padding piano roll with 0s.')
                R = np.pad(R, ((0, 0), (0, S.shape[1] - R.shape[1])), constant_values=0)
            elif S.shape[1] < R.shape[1]:
                print(f'Shape mismatch: S={S.shape[1]} R={R.shape[1]}. Cutting piano roll\'s tail.')
                R = R[:, :S.shape[1]]
            assert R.shape[1] == S.shape[1]

            frames_dict[piece_name]['x'] = S
            frames_dict[piece_name]['y'] = R

        return frames_dict

    def get_frames(self, piece, first_n_sec=0, torchify=True):
        frames = self.frames_dict[piece]['x']
        if first_n_sec:
            sec2frame = self.sr / self.hop_length
            frame_to = int(np.round(first_n_sec * sec2frame))
            frames = frames[:, :frame_to].copy()
        return torch.tensor(frames).float() if torchify else frames

    def get_pianoroll(self, piece, first_n_sec=0):
        pianoroll = self.frames_dict[piece]['y']
        if first_n_sec:
            sec2frame = self.sr / self.hop_length
            frame_to = int(np.round(first_n_sec * sec2frame))
            pianoroll = pianoroll[:, :frame_to].copy()
        return pianoroll

    def plot_frames(self, piece, figsize, first_n_sec=0):
        from librosa.display import specshow
        import matplotlib.pyplot as plt

        frames = self.get_frames(piece, first_n_sec=first_n_sec, torchify=False)
        fig = plt.figure(figsize=figsize)
        specshow(frames, sr=self.sr, hop_length=self.hop_length, cmap='Greys')
        plt.title(f'MUS | {self.instrument} | {piece}')
        return fig

    def plot_pianoroll(self, piece, figsize, first_n_sec=0, limit_range=False, limits=None):
        from librosa.display import specshow
        import matplotlib.pyplot as plt

        pianoroll_full = self.get_pianoroll(piece, first_n_sec=first_n_sec)

        if limit_range:
            if limits is not None and len(limits) == 2:
                min_notes, max_notes = limits
            else:
                noteset = np.argwhere(pianoroll_full.sum(axis=1))
                min_notes = noteset.min()
                max_notes = noteset.max()
            pianoroll = pianoroll_full[min_notes:max_notes+1]
        else:
            pianoroll = pianoroll_full

        fig = plt.figure(figsize=figsize)
        ax = plt.gca(); ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False);
        specshow(pianoroll, x_axis='frames', y_axis='frames', cmap='Greys');
        plt.title(f'MUS | {self.instrument} | {piece}')
        return fig


class SMDPieceFramesFactory(object):
    def __init__(self, piece, datadir='data/SMD/',
                 sr=22050, n_fft=512, hop_length=512//4, log_str=10000,
                 keep_bins=0, note_range_limit=None):
        config_string = f'{sr}_{n_fft}_{hop_length}_{log_str}_{keep_bins}'
        config_savepath = f'data/{config_string}/SMD_{piece}.npy'

        if not os.path.isfile(config_savepath):
            frames = self.load_piece(
                piece, datadir, sr, n_fft, hop_length, log_str, keep_bins)
            os.makedirs(f'data/{config_string}', exist_ok=True)
            np.save(config_savepath, frames)
        else:
            frames = np.load(config_savepath, allow_pickle=True)[np.newaxis][0]

        file_mid = find_files(datadir, f'{piece}.mid')[0]
        self.mid = pm.PrettyMIDI(file_mid)

        if note_range_limit is not None:
            assert len(note_range_limit) == 2
            note_min, note_max = note_range_limit
            notes_present = np.unique(np.concatenate([[n.pitch for n in i.notes] for i in self.mid.instruments]))
            assert min(notes_present) >= note_min, f'{min(notes_present)} >= {note_min}'
            assert max(notes_present) <= note_max, f'{max(notes_present)} >= {note_max}'

        self.piece_name = piece
        self.config_string = config_string
        self.frames = frames

        self.sr = sr
        self.hop_length = hop_length

    def load_piece(self, piece, datadir,
                   sr=22050, n_fft=512, hop_length=512//4,
                   log_str=10000, keep_bins=0):
        file = find_files(datadir, f'{piece}.mp3')[0]

        y, _ = librosa.load(file, sr=sr)

        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        S, _ = librosa.core.magphase(S[1:, :]) # discard lowest (DC) bin (1+n_fft//2 --> n_fft//2)
        S = spec_norm(S, n_fft, log=log_str>0, strength=log_str)

        if keep_bins:
            S = S[:keep_bins, :] # low-pass (discard higher frequency bins)

        return S

    def get_frames(self, keep_frames=0, torchify=True):
        S = self.frames
        if keep_frames:
            S = S[:, :keep_frames]
        return torch.tensor(S).float() if torchify else S

    def get_pianoroll(self, keep_frames=0, limit_range=False):
        pianoroll = self.mid.get_piano_roll(fs=float(self.sr/self.hop_length))

        # 0-pad the labels on the right to account for release part of ADSR of final note(s)
        T_S = self.frames.shape[1] # snippet's number of time frames
        T_P = pianoroll.shape[1] # piano-roll's number of time frames
        if T_S > T_P:
            pianoroll = np.pad(pianoroll, ((0, 0), (0, T_S - T_P)), constant_values=0)
        elif T_S < T_P:
            pianoroll = pianoroll[:, :T_S]
        if keep_frames:
            pianoroll = pianoroll[:, :keep_frames]
        return pianoroll

    def plot_frames(self, figsize, keep_frames=0):
        from librosa.display import specshow
        import matplotlib.pyplot as plt

        frames = self.get_frames(keep_frames=keep_frames, torchify=False)
        fig = plt.figure(figsize=figsize)
        specshow(frames, sr=self.sr, hop_length=self.hop_length, cmap='Greys')
        plt.title(f'{self.piece_name} | {keep_frames} | {self.config_string}')
        return fig

    def plot_pianoroll(self, figsize, keep_frames=0, limit_range=False, limits=None):
        from librosa.display import specshow
        import matplotlib.pyplot as plt

        pianoroll = self.get_pianoroll(keep_frames=keep_frames)

        if limit_range:
            if limits is not None and len(limits) == 2:
                min_notes, max_notes = limits
            else:
                noteset = np.argwhere(pianoroll.sum(axis=1))
                min_notes = noteset.min()
                max_notes = noteset.max()
            pianoroll = pianoroll[min_notes:max_notes+1]

        fig = plt.figure(figsize=figsize)
        ax = plt.gca(); ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False);
        specshow(pianoroll, x_axis='frames', y_axis='frames', cmap='Greys');
        plt.title(f'{self.piece_name} | {keep_frames} | {self.config_string}')
        return fig
