import argparse
import os

import numpy as np
import pretty_midi as pm
import torch
import torchvision
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from data import MultiNoteTimbreFramesFactory, MAPS_ISOL_NoteFrames_MultiSource, JitterTransform
from test_tube import HyperOptArgumentParser
from glow import ConditionalGlow
from visualizers import plot_frames

MAX_EPOCHS = 2500


def get_arguments():
    '''Parses script arguments.'''

    parser = argparse.ArgumentParser(
        description='Trains A Conditional NoteGlow model for each note from the specified range.')
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


class ConditionalGlowFramesTraining(ConditionalGlow):
    def __init__(self, note_range, hp, dataset, *args):
        super().__init__(*args)
        self.validation_losses = []

        zn = self.prior.sample((self.hparams.num_classes,))
        zs = torch.eye(self.hparams.num_classes).to(zn.device)
        self.sample_batch = torch.cat([zs, zn], dim=1)
        self.xlabels = [pm.note_number_to_name(i) for i in range(pm.note_name_to_number(note_range[0]),
                                                                 pm.note_name_to_number(note_range[1])+1)]

        noterange_dirname = f'{pm.note_name_to_number(note_range[0]):03}_{pm.note_name_to_number(note_range[1]):03}' + \
                            f'_{note_range[0]}_{note_range[1]}'
        hyperparam_config = hp.config.split('/')[-1].split('.')[0]
        self.savepath = f'logs/{dataset}/{hyperparam_config}/{noterange_dirname}/'
        os.makedirs(self.savepath, exist_ok=True)

    def cuda(self, device=None):
        super().cuda(device=device)
        self.sample_batch = self.sample_batch.to(self.device)
        return self

    def to(self, device=None, dtype=None, non_blocking=False):
        super().to(device=device, dtype=dtype, non_blocking=non_blocking)
        self.sample_batch = self.sample_batch.to(self.device)
        return self

    def on_train_epoch_start(self):
        self.grad_norms = []

        images = self.inverse(self.sample_batch)
        fig_samples = plot_frames(images.cpu(), len(images), xlabels=self.xlabels, verbose=False);
        self.logger.experiment.add_figure('samples_raw', fig_samples, global_step=self.trainer.current_epoch)

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        grad_norm = torch.cat([p.grad.view(-1) for p in self.chain.parameters()]).norm().item()
        self.grad_norms.append(grad_norm)

    def training_epoch_end(self, training_step_outputs):
        train_loss = np.mean([item['loss'].item() for item in training_step_outputs])
        train_nll = np.mean([item['nll'].item() for item in training_step_outputs])
        train_l2 = np.mean([item['l2'].item() for item in training_step_outputs])
        train_log_det = np.mean([item['log_det'].item() for item in training_step_outputs])

        mse_zs = np.mean([item['mse_zs'].item() for item in training_step_outputs])
        neg_ce_nc_zn = np.mean([item['neg_ce_nc_zn'].item() for item in training_step_outputs])
        nc_loss_last = np.mean([item['nc_loss_last'].item() for item in training_step_outputs])

        self.logger.experiment.add_scalar('train_loss', train_loss, self.trainer.current_epoch)
        self.logger.experiment.add_scalar('train_nll', train_loss, self.trainer.current_epoch)
        self.logger.experiment.add_scalar('train_l2', train_l2, self.trainer.current_epoch)
        self.logger.experiment.add_scalar('train_log_det', train_log_det, self.trainer.current_epoch)

        self.logger.experiment.add_scalar('mse_zs', mse_zs, self.trainer.current_epoch)
        self.logger.experiment.add_scalar('neg_ce_nc_zn', neg_ce_nc_zn, self.trainer.current_epoch)
        self.logger.experiment.add_scalar('nc_loss_last', nc_loss_last, self.trainer.current_epoch)

        self.logger.experiment.add_scalar('grad_norm_mean', np.mean(self.grad_norms), self.trainer.current_epoch)
        self.logger.experiment.add_scalar('grad_norm_max', max(self.grad_norms), self.trainer.current_epoch)

        from pytorch_lamb import log_lamb_rs
        log_lamb_rs(self.optimizer_main, self.logger.experiment, self.trainer.current_epoch)

    def validation_epoch_end(self, val_step_outputs):
        val_loss = np.mean([item['val_loss'].item() for item in val_step_outputs])
        self.logger.experiment.add_scalar('val_loss', val_loss, self.trainer.current_epoch)
        self.validation_losses.append(val_loss)

    def on_train_end(self):
        # save hyperparams and metrics
        hparam_dict = dict(self.hparams)
        metric_dict = {'val_loss_mean': np.mean(self.validation_losses),
                       'val_loss_best': min(self.validation_losses)}
        self.logger.experiment.add_hparams(hparam_dict, metric_dict)

    def configure_optimizers(self):
        from pytorch_lamb import Lamb

        self.optimizer_main = Lamb(self.chain.parameters(), lr=self.hparams.lr)
        self.optimizer_nc = torch.optim.Adam(self.nuisance_classifier.parameters(), lr=self.hparams.lr)
        return self.optimizer_main, self.optimizer_nc


def main():
    args = get_arguments()

    available_notes = range(pm.note_name_to_number('A0'), pm.note_name_to_number('C8')+1)
    assert pm.note_name_to_number(args.note_from) in available_notes
    assert pm.note_name_to_number(args.note_to) in available_notes
    assert pm.note_name_to_number(args.note_from) <= pm.note_name_to_number(args.note_to)
    assert os.path.exists(f'config/{args.hp}.json')
    assert args.dataset in ['KS', 'MAPS', 'MAPS_R', 'MAPS_R+KS']
    print('run arguments:', args.note_from, args.note_to, args.hp, args.dataset)

    # load hyperparam config
    parser = HyperOptArgumentParser()
    parser.json_config('--config', default=f'config/{args.hp}.json')
    hp = parser.parse_args({})

    note_range = (args.note_from, args.note_to)

    # prepare datasets
    transform_train, transform_valid = get_data_transforms(hp)

    if args.dataset == 'KS':
        dataset_factory = MultiNoteTimbreFramesFactory(
            note_range, 'data/KeyScapes/notes', sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length,
            log_str=hp.log_str, keep_bins=hp.keep_bins, keep_frames=hp.keep_frames)

        train_ds = dataset_factory.make_dataset(shuffle=True, split='train', transform=transform_train)
        valid_ds = dataset_factory.make_dataset(shuffle=True, split='valid', transform=transform_valid)
        test_ds = dataset_factory.make_dataset(shuffle=True, split='valid', transform=transform_valid)

    elif args.dataset == 'MAPS':
        train_ds = MAPS_ISOL_NoteFrames_MultiSource(note_range, mapsdir='data/MAPS/',
            sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length, log_str=hp.log_str,
            keep_bins=hp.keep_bins, keep_frames=hp.keep_frames,
            shuffle=True, split='train', transform=transform_train)
        valid_ds = MAPS_ISOL_NoteFrames_MultiSource(note_range, mapsdir='data/MAPS/',
            sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length,
            log_str=hp.log_str, keep_bins=hp.keep_bins, keep_frames=hp.keep_frames,
            shuffle=True, split='valid', transform=transform_valid)
        test_ds = MAPS_ISOL_NoteFrames(note_name=note_name, mapsdir='data/MAPS/',
            sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length,
            log_str=hp.log_str, keep_bins=hp.keep_bins, keep_frames=hp.keep_frames,
            shuffle=True, split='valid', transform=transform_valid)

    elif args.dataset == 'MAPS_R':
        train_ds = MAPS_ISOL_NoteFrames_MultiSource(note_range, mapsdir='data/MAPS/',
            sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length, log_str=hp.log_str,
            keep_bins=hp.keep_bins, keep_frames=hp.keep_frames,
            shuffle=True, split='train', realistic=True, transform=transform_train)
        valid_ds = MAPS_ISOL_NoteFrames_MultiSource(note_range, mapsdir='data/MAPS/',
            sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length, log_str=hp.log_str,
            keep_bins=hp.keep_bins, keep_frames=hp.keep_frames,
            shuffle=True, split='valid', realistic=True, transform=transform_valid)
        test_ds = MAPS_ISOL_NoteFrames_MultiSource(note_range, mapsdir='data/MAPS/',
            sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length, log_str=hp.log_str,
            keep_bins=hp.keep_bins, keep_frames=hp.keep_frames,
            shuffle=False, split='test', realistic=True, transform=transform_valid)

    elif args.dataset == 'MAPS_R+KS':
        train_ds = MAPS_ISOL_NoteFrames_MultiSource(note_range, mapsdir='data/MAPS/',
            sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length, log_str=hp.log_str,
            keep_bins=hp.keep_bins, keep_frames=hp.keep_frames,
            shuffle=True, split='train', realistic=True, transform=transform_train)
        valid_ds = MAPS_ISOL_NoteFrames_MultiSource(note_range, mapsdir='data/MAPS/',
            sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length, log_str=hp.log_str,
            keep_bins=hp.keep_bins, keep_frames=hp.keep_frames,
            shuffle=True, split='valid', realistic=True, transform=transform_valid)
        test_ds = MAPS_ISOL_NoteFrames_MultiSource(note_range, mapsdir='data/MAPS/',
            sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length,
            log_str=hp.log_str, keep_bins=hp.keep_bins, keep_frames=hp.keep_frames,
            shuffle=False, split='test', realistic=True, transform=transform_valid)
        # augment with KeyScapes data
        dataset_factory = MultiNoteTimbreFramesFactory(
            note_range, 'data/KeyScapes/notes', sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length,
            log_str=hp.log_str, keep_bins=hp.keep_bins, keep_frames=hp.keep_frames)
        train_ds_KS = dataset_factory.make_dataset(shuffle=True, split='full')
        train_ds.x = torch.cat([train_ds.x, train_ds_KS.x], dim=0)
        train_ds.y = torch.cat([train_ds.y, train_ds_KS.y], dim=0)

    print(f'Note range {note_range} | train/valid/test sizes: ' +
          f'{train_ds.x.shape[0]}/{valid_ds.x.shape[0]}/{test_ds.x.shape[0]}')

    train_dl = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True, num_workers=hp.num_workers, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=hp.batch_size*2, num_workers=hp.num_workers, pin_memory=True)
    #test_dl = DataLoader(test_ds, batch_size=hp.batch_size*2, num_workers=hp.num_workers, pin_memory=True)

    # instantiate model
    num_classes = train_ds.y.unique().size(0)
    data_dim = hp.n_fft // 2
    assert data_dim == train_ds.x.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ConditionalGlowFramesTraining(
        note_range, hp, args.dataset,
        num_classes, data_dim, hp.num_blocks, hp.acl_arch, data_dim * hp.mlp_width_factor, hp.mlp_depth_layers,
        hp.cnn_kernel_size, hp.cnn_stride, hp.cnn_channels, hp.cnn_depth, hp.cnn_dilated,
        hp.actf, hp.dropout, hp.weight_norm, hp.l2_reg_str, hp.learning_rate,
        hp.permutation, hp.use_actnorm, hp.LU_decomposed, hp.from_log_s,
        hp.nc_width, hp.sem_mse, hp.nui_ce
    ).to(device)

    # instantiate trainer with appropriate callbacks
    tensorboard_logger = pl.loggers.TensorBoardLogger(
        model.savepath, name='', version='', default_hp_metric=False)
    #tb_logger = pl.loggers.TensorBoardLogger(nvp.savepath, name='', version='', default_hp_metric=False)

    valid_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=tensorboard_logger.log_dir, filename='valid',#+f'-{epoch:03d}-{val_loss:.4f}',
        monitor='val_loss', verbose=False, save_top_k=1, mode='min')

    train_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=tensorboard_logger.log_dir, filename='train',#+f'-{epoch:03d}-{val_loss:.4f}',
        monitor='train_loss', verbose=False, save_top_k=1, mode='min')

    earlystopp_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=-hp.num_epochs, verbose=False)

    trainer = pl.Trainer(
        gpus=[0],
        check_val_every_n_epoch=1,
        logger=tensorboard_logger,
        callbacks=[valid_checkpoint_callback, train_checkpoint_callback] + ([] if hp.num_epochs >= 0 else [earlystopp_callback]),
        max_epochs=hp.num_epochs if hp.num_epochs >= 0 else MAX_EPOCHS
    )

    print(tensorboard_logger.log_dir)

    if os.path.exists(f'{model.savepath}/lockfile'):
        print(f'Lockfile {model.savepath}/lockfile already exists. Skipping trained configuration.')
    else:
        f = open(f'{model.savepath}/lockfile', 'x')
        f.close()
        trainer.fit(model, train_dl, valid_dl)

        # plot samples of best checkpoint (as per validation loss) after training
        ckpt_best = trainer.checkpoint_callback.best_model_path # f'{model.savepath}/valid.ckpt'
        model_best = ConditionalGlow.load_from_checkpoint(ckpt_best)

        images = model_best.inverse(model.sample_batch.cpu())
        fig_samples = plot_frames(images, len(images), xlabels=model.xlabels, verbose=False);
        model.logger.experiment.add_figure('samples_raw', fig_samples, global_step=None)


if __name__ == '__main__':
    main()
