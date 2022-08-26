import argparse
import os

import numpy as np
import pretty_midi as pm
import torch
import torchvision
import pytorch_lightning as pl

from data import SingleNoteTimbreFramesFactory, MAPS_ISOL_NoteFrames, JitterTransform
from test_tube import HyperOptArgumentParser
from flow import Flow

MAX_EPOCHS = 1000


def get_arguments():
    '''Parses script arguments.'''

    parser = argparse.ArgumentParser(
        description='Trains 1 NoteFlow model for each note from the specified range.')
    parser.add_argument(
        '--note_from', type=str, required=True,
        help='Name of the lower bound note. Available range is the classical piano range A0-C8.')
    parser.add_argument(
        '--note_to', type=str, required=True,
        help='Name of the upper bound note. Available range is the classical piano range A0-C8.')
    parser.add_argument(
        '--hp', type=str, required=True,
        help='Name of the hyperparameter configuration json file without extension,')
    parser.add_argument(
        '--dataset', type=str, required=True,
        help='Dataset to use for training. Options include KS, MAPS, MAPS_R and MAPS_R+KS.')
    return parser.parse_args()


def get_data_transforms(hp):
    transform_train = []; transform_valid = [];

    if hp.jitter_scale:
        transform_train.append(JitterTransform(amount=hp.jitter_scale))
    if hp.data_shift:
        transform_train.append(torchvision.transforms.Lambda(lambda x: x * hp.data_scale + hp.data_shift))
        transform_valid.append(torchvision.transforms.Lambda(lambda x: x * hp.data_scale + hp.data_shift))

    transform_train = torchvision.transforms.Compose(transform_train) if len(transform_train) > 0 else None
    transform_valid = torchvision.transforms.Compose(transform_valid) if len(transform_valid) > 0 else None

    return (transform_train, transform_valid)


class FlowFrames(Flow):
    def __init__(self, note_number, note_name, hp, dataset, *args):
        super().__init__(*args)
        note_dirname = f'{note_number:03}_{note_name}'
        hyperparam_config = hp.config.split('/')[-1].split('.')[0]
        self.savepath = f'logs/{dataset}/{hyperparam_config}/{note_dirname}/'
        os.makedirs(self.savepath, exist_ok=True)


def main():
    args = get_arguments()

    available_notes = range(pm.note_name_to_number('A0'), pm.note_name_to_number('C8')+1)
    assert pm.note_name_to_number(args.note_from) in available_notes
    assert pm.note_name_to_number(args.note_to) in available_notes
    assert pm.note_name_to_number(args.note_from) <= pm.note_name_to_number(args.note_to)
    assert os.path.exists(f'config/{args.hp}.json')
    assert args.dataset in ['KS', 'MAPS', 'MAPS_R', 'MAPS_R+KS']
    print('run arguments:', args.note_from, args.note_to, args.hp, args.dataset)
    # exit()

    # load hyperparam config
    parser = HyperOptArgumentParser()
    parser.json_config('--config', default=f'config/{args.hp}.json')
    hp = parser.parse_args({})

    for note_number in range(pm.note_name_to_number(args.note_from), pm.note_name_to_number(args.note_to)+1):
        note_name = pm.note_number_to_name(note_number)

        # prepare datasets
        transform_train, transform_valid = get_data_transforms(hp)

        if args.dataset == 'KS':
            dataset_factory = SingleNoteTimbreFramesFactory(
                note_name, 'data/KeyScapes/notes', sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length,
                log_str=hp.log_str, keep_bins=hp.keep_bins, keep_frames=hp.keep_frames)

            train_ds = dataset_factory.make_dataset(shuffle=True, split='train', transform=transform_train)
            valid_ds = dataset_factory.make_dataset(shuffle=True, split='valid', transform=transform_valid)
            test_ds = dataset_factory.make_dataset(shuffle=True, split='valid', transform=transform_valid)

        elif args.dataset == 'MAPS':
            train_ds = MAPS_ISOL_NoteFrames(note_name=note_name, mapsdir='data/MAPS/',
                sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length,
                log_str=hp.log_str, keep_bins=hp.keep_bins, keep_frames=hp.keep_frames,
                shuffle=True, split='train', transform=transform_train)
            valid_ds = MAPS_ISOL_NoteFrames(note_name=note_name, mapsdir='data/MAPS/',
                sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length,
                log_str=hp.log_str, keep_bins=hp.keep_bins, keep_frames=hp.keep_frames,
                shuffle=True, split='valid', transform=transform_valid)
            test_ds = MAPS_ISOL_NoteFrames(note_name=note_name, mapsdir='data/MAPS/',
                sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length,
                log_str=hp.log_str, keep_bins=hp.keep_bins, keep_frames=hp.keep_frames,
                shuffle=True, split='valid', transform=None)

        elif args.dataset == 'MAPS_R':
            train_ds = MAPS_ISOL_NoteFrames(note_name=note_name, mapsdir='data/MAPS/',
                sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length,
                log_str=hp.log_str, keep_bins=hp.keep_bins, keep_frames=hp.keep_frames,
                shuffle=True, split='train', realistic=True, transform=transform_train)
            valid_ds = MAPS_ISOL_NoteFrames(note_name=note_name, mapsdir='data/MAPS/',
                sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length,
                log_str=hp.log_str, keep_bins=hp.keep_bins, keep_frames=hp.keep_frames,
                shuffle=True, split='valid', realistic=True, transform=transform_valid)
            test_ds = MAPS_ISOL_NoteFrames(note_name=note_name, mapsdir='data/MAPS/',
                sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length,
                log_str=hp.log_str, keep_bins=hp.keep_bins, keep_frames=hp.keep_frames,
                shuffle=False, split='test', realistic=True, transform=None)

        elif args.dataset == 'MAPS_R+KS':
            train_ds = MAPS_ISOL_NoteFrames(note_name=note_name, mapsdir='data/MAPS/',
                sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length,
                log_str=hp.log_str, keep_bins=hp.keep_bins, keep_frames=hp.keep_frames,
                shuffle=True, split='train', realistic=True, transform=transform_train)
            valid_ds = MAPS_ISOL_NoteFrames(note_name=note_name, mapsdir='data/MAPS/',
                sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length,
                log_str=hp.log_str, keep_bins=hp.keep_bins, keep_frames=hp.keep_frames,
                shuffle=True, split='valid', realistic=True, transform=transform_valid)
            test_ds = MAPS_ISOL_NoteFrames(note_name=note_name, mapsdir='data/MAPS/',
                sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length,
                log_str=hp.log_str, keep_bins=hp.keep_bins, keep_frames=hp.keep_frames,
                shuffle=False, split='test', realistic=True, transform=None)
            # augment with KeyScapes data
            dataset_factory = SingleNoteTimbreFramesFactory(
                note_name, 'data/KeyScapes/notes', sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length,
                log_str=hp.log_str, keep_bins=hp.keep_bins, keep_frames=hp.keep_frames)
            train_ds_KS = dataset_factory.make_dataset(shuffle=True, split='full')
            train_ds.x = torch.cat([train_ds.x, train_ds_KS.x], dim=0)

        print(f'Note {note_number:03} {note_name} | train/valid/test sizes: ' +
              f'{train_ds.x.shape[0]}/{valid_ds.x.shape[0]}/{test_ds.x.shape[0]}')

        # instantiate model
        data_dim = train_ds.x.shape[1]
        nvp = FlowFrames(
            note_number, note_name, hp, args.dataset,
            data_dim,
            hp.num_blocks,
            hp.mlp_width,
            hp.mlp_depth,
            hp.mlp_actf,
            hp.weight_norm,
            hp.permutation,
            hp.learning_rate,
            hp.l2_reg_str,
            hp.dropout,
            hp.batch_size,
            hp.num_workers,
            train_ds,
            valid_ds,
            test_ds
        )

        # instantiate trainer with appropriate callbacks
        tb_logger = pl.loggers.TensorBoardLogger(nvp.savepath, name='', version='', default_hp_metric=False)
        callbacks = [pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', dirpath=nvp.savepath, filename='best')]
        callbacks += [] if hp.num_epochs >= 0 else [pl.callbacks.EarlyStopping(monitor='val_loss', patience=-hp.num_epochs)]

        trainer = pl.Trainer(
            gpus=[0],
            gradient_clip_val=5,
            logger=tb_logger,
            callbacks=callbacks,
            checkpoint_callback=True,
            max_epochs=hp.num_epochs if hp.num_epochs >= 0 else MAX_EPOCHS
        )

        if os.path.exists(f'{nvp.savepath}/lockfile'):
            print(f'Lockfile {nvp.savepath}/lockfile already exists. Skipping trained configuration.')
        else:
            trainer.fit(nvp)
            f = open(f'{nvp.savepath}/lockfile', 'x')
            f.close()


if __name__ == '__main__':
    main()
