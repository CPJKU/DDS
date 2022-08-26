import argparse
import os

import numpy as np
import pretty_midi as pm
import torch
import torch.optim as optim
import torchvision
import pytorch_lightning as pl

import matplotlib.pyplot as plt
import librosa
import librosa.display
import mir_eval

from data import MAPS_ISOL_NoteFrames, MAPS_MUS_PieceFramesFactory, SingleNoteTimbreFramesFactory
from test_tube import HyperOptArgumentParser
from visualizers import (get_figsize, plot_matrix, plot_reconstruction_performance, plot_eval,
                         plot_pr_roc_curves, save_fig_safely)
from flow import Flow
from glow import ConditionalGlow
from nmf import PyTorchNMF
from flownmf import FlowNMF
from flownmf2 import FlowNMF2
from flownmf3 import FlowNMF3
from misc import find_files, inceptdict
from metrics import get_optimal_threshold, calc_metrics, calc_aucs


def get_arguments():
    '''Parses script arguments.'''

    parser = argparse.ArgumentParser(
        description='Evaluates specified method on specified piece from MAPS/MUS.')
    parser.add_argument(
        '--method', type=str, required=True,
        help='Name of the method to test. Options are [NMF, DDSv1, DDSv2, DDSv3].')
    parser.add_argument(
        '--hp', type=str, required=True,
        help='Name of the hyperparameter configuration json file without extension,')
    parser.add_argument(
        '--piece', type=int, default=0,
        help='Number of the piece to decompose (in range [0; 29]).')
    parser.add_argument(
        '--first_n_sec', type=int, default=30,
        help='The initial number of seconds of the piece to decompose. 0 for full length.')
    parser.add_argument(
        '--max_iter', type=int, default=1000,
        help='Number of iterations to do with each method decomposition.')
    parser.add_argument(
        '--max_frames', type=int, default=0,
        help='Number of time frames to keep from the input snippet for decomposition. 0 for all of them')
    parser.add_argument(
        '--n_cps', type=int, default=20,
        help='Number of components to use with DDSv2/3.')
    parser.add_argument(
        '--dds_step_size', type=float, default=0.001,
        help='The step size (learning rate) for the DDS decomposition optimizer.')
    parser.add_argument(
        '--instr', type=int, default=8,
        help='Which instrument from the MAPS set to test on data from.')
    parser.add_argument(
        '--train_set', type=str, default='MAPS_R',
        help='Dataset used for method initialization. Options include KS, MAPS, MAPS_R and MAPS_R+KS.')
    parser.add_argument(
        '--gpu_id', type=int, default=0,
        help='Which GPU to use. Options are [0, 1, ..., N-1] on N-GPU machines.')
    parser.add_argument(
        '--csvfile', type=str, default='results_grid_MAPS_MUS',
        help='Name of the csv file (without extension) where the results of this test are appended.')
    return parser.parse_args()


def main():
    args = get_arguments()

    assert args.method in ['NMF', 'DDSv1', 'DDSv2', 'DDSv3']
    assert os.path.exists(f'config/{args.hp}.json')
    assert args.piece in range(30) # there's exactly 30 classical piano pieces in MAPS/MUS
    assert args.first_n_sec >= 0
    assert args.max_iter > 0
    assert args.max_frames >= 0
    assert args.n_cps >= 1
    assert args.instr in range(9)
    assert args.train_set in ['KS', 'MAPS', 'MAPS_R', 'MAPS_R+KS']
    assert args.gpu_id >= 0
    print('run arguments:', args.__dict__)

    # load hyperparam config
    parser = HyperOptArgumentParser()
    parser.json_config('--config', default=f'config/{args.hp}.json')
    hp = parser.parse_args({})

    # get MAPS/MUS data factory
    mus_piece_factory = MAPS_MUS_PieceFramesFactory(
        instr=args.instr, mapsdir='data/MAPS/',
        sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length,
        log_str=hp.log_str, keep_bins=hp.keep_bins,
        note_range_limit=(pm.note_name_to_number('C2'), pm.note_name_to_number('C7')))

    mus_pieces = sorted(list(mus_piece_factory.frames_dict.keys()))
    piece = mus_pieces[args.piece]

    # produce data to evaluate on
    S = mus_piece_factory.get_frames(piece, first_n_sec=args.first_n_sec)
    pianoroll = mus_piece_factory.get_pianoroll(piece, first_n_sec=args.first_n_sec)
    if args.max_frames:
        S = S[:, :args.max_frames]
        pianoroll = pianoroll[:, :args.max_frames]

    # determine log path for this config
    logdir = f'logs/sede/{args.hp}/{args.train_set}/MUS_{mus_piece_factory.instrument}/{piece}/{args.method}'
    logdir += f'/first_{args.first_n_sec:03}_s' if args.first_n_sec else '/full'

    # extract used note range
    note_range_low, note_range_high = [int(n) for n in 'M36-95'.replace('M', '').split('-')]
    note_list = list(range(note_range_low, note_range_high+1))

    # load used note datasets
    train_ds = {}

    for note in note_list:
        if 'KS' in args.train_set:
            KS_factory = SingleNoteTimbreFramesFactory(pm.note_number_to_name(note),
                'data/KeyScapes/notes', sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length,
                log_str=hp.log_str, keep_bins=hp.keep_bins, keep_frames=hp.keep_frames)

        if args.train_set == 'KS':
            ds = KS_factory.make_dataset(shuffle=False, split='full')

        elif args.train_set == 'MAPS':
            ds = MAPS_ISOL_NoteFrames(
                note_name=pm.note_number_to_name(note), mapsdir='data/MAPS/',
                sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length,
                log_str=hp.log_str, keep_bins=hp.keep_bins, keep_frames=hp.keep_frames,
                shuffle=False, split='full', transform=None)

        elif 'MAPS_R' in args.train_set:
            ds = MAPS_ISOL_NoteFrames(
                note_name=pm.note_number_to_name(note), mapsdir='data/MAPS/',
                sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.hop_length,
                log_str=hp.log_str, keep_bins=hp.keep_bins, keep_frames=hp.keep_frames,
                shuffle=False, split='full', realistic=True, transform=None)

            # conditionally augment with KeyScapes data
            if args.train_set == 'MAPS_R+KS':
                ds_KS = KS_factory.make_dataset(shuffle=False, split='full')
                ds.x = torch.cat([ds.x, ds_KS.x], dim=0)

        train_ds[note] = ds.x.T.numpy()

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # load NoteFlow(s) if method is a variant of DDS
    if 'DDS' in args.method:
        if 'DDSv3' == args.method:
            note_range = ('C2', 'C7')
            noterange_dirname = f'{pm.note_name_to_number(note_range[0]):03}_' + \
                                f'{pm.note_name_to_number(note_range[1]):03}_' + \
                                f'{note_range[0]}_{note_range[1]}'

            savepath = f'logs/{args.train_set}/{args.hp}/{noterange_dirname}/checkpoints'
            filepath = find_files(savepath, '*.ckpt')[0]

            flow = ConditionalGlow.load_from_checkpoint(filepath).to(device)
        else:
            flows = {}
            for note in note_list:
                kwargs = {
                    'data_dim': train_ds[note].shape[0], 'blocks': hp.num_blocks, 'width': hp.mlp_width,
                    'depth': hp.mlp_depth, 'actf': hp.mlp_actf, 'norm': hp.weight_norm,
                    'perm': hp.permutation, 'lr': hp.learning_rate, 'l2str': hp.l2_reg_str,
                    'drop': hp.dropout, 'bs': hp.batch_size, 'num_workers': hp.num_workers,
                    'train_ds': None, 'valid_ds': None, 'test_ds': None
                }
                note_dirname = f'{note:03}_{pm.note_number_to_name(note)}'
                #hyperparam_config = hp.config.split('/')[-1].split('.')[0]
                savepath = f'logs/{args.train_set}/{args.hp}/{note_dirname}/'
                filepath = find_files(savepath, 'best*.ckpt')[0]

                flows[note] = Flow.load_from_checkpoint(filepath, **kwargs).cuda(args.gpu_id)

            flows_dict = {note: flow for note, flow in flows.items()} # multi-source DDS

    # configure logging in a method-specific manner
    if args.method == 'NMF':
        W_norm = False # just for correct logging

        logdir_full = f'{logdir}/it_{args.max_iter:05d}_Wn_{W_norm}'

    elif args.method == 'DDSv1':
        lr = args.dds_step_size; lw = 0.0001; Z_init = 'zero'; H_init = 'random'; Wn_rescale = True;
        lr_str = '_'.join(f'{lr:.3f}'.split('.'))
        lw_str = '_'.join(f'{lw:.5f}'.split('.'))

        logdir_full = f'{logdir}/it_{args.max_iter:05d}_lr_{lr_str}_lw_{lw_str}_Zi_{Z_init}_Hi_{H_init}'

    elif args.method == 'DDSv2' or args.method == 'DDSv3':
        lr = args.dds_step_size; lw = 0.0001; Z_init = 'zero'; H_init = 'random'; Wn_rescale = True;
        lr_str = '_'.join(f'{lr:.3f}'.split('.'))
        lw_str = '_'.join(f'{lw:.5f}'.split('.'))

        logdir_full = f'{logdir}/it_{args.max_iter:05d}_cps_{args.n_cps:03d}_lr_{lr_str}_lw_{lw_str}_Zi_{Z_init}_Hi_{H_init}'

    # method-agnostic part of logging setup
    logdir_full += f'_keep_{args.max_frames:05d}' if args.max_frames else ''
    os.makedirs(logdir_full, exist_ok=True)
    print(logdir_full)

    # produce H_hat in a method-specific manner
    if args.method == 'NMF':
        # re-store result if previously computed for this config
        if not os.path.isfile(f'{logdir_full}/X.npy'):
            W_init = np.concatenate(list(train_ds.values()), axis=1)

            nmf = PyTorchNMF(S, W_init.shape[1], W_init=W_init, W_norm=W_norm, H_init=1.0, optimizer=optim.Adam)
            W, H = nmf.fit(max_iter=args.max_iter, W_fixed=True)

            X = W @ H
            S = S.numpy()
            R = np.abs(S - X)

            np.save(f'{logdir_full}/X.npy', X)
            np.save(f'{logdir_full}/W.npy', W)
            np.save(f'{logdir_full}/H.npy', H)
            np.save(f'{logdir_full}/R.npy', R)
            np.save(f'{logdir_full}/S.npy', S)
        else:
            print(f'Loading pre-computed results from {logdir_full}')
            X = np.load(f'{logdir_full}/X.npy')
            W = np.load(f'{logdir_full}/W.npy')
            H = np.load(f'{logdir_full}/H.npy')
            R = np.load(f'{logdir_full}/R.npy')
            S = np.load(f'{logdir_full}/S.npy')

        sources_samples = [a.shape[1] for a in train_ds.values()]
        sources_end = np.cumsum(sources_samples)
        sources_start = sources_end - sources_end[0]

        if not W_norm:
            W_norms = np.linalg.norm(W, axis=0)
            W_norms[W_norms == 0] = 1 # avoid NaNs later on, some W entries are 0-vectors..
            H_hat = [(W_norms[start: end, np.newaxis] * H[start: end]).sum(0) for start, end in zip(sources_start, sources_end)]
            H_hat = np.concatenate([act[np.newaxis] for act in H_hat])
        else:
            H_hat = [H[start: end].sum(0) for start, end in zip(sources_start, sources_end)]
            H_hat = np.concatenate([act[np.newaxis] for act in H_hat])

    elif args.method == 'DDSv1':
        # re-store result if previously computed for this config
        if not os.path.isfile(f'{logdir_full}/X.npy'):
            flow_nmf = FlowNMF(S, flows_dict, nll_weight=lw, Z_init=Z_init, H_init=H_init, lr=lr).to(device)
            flow_nmf.fit(max_iter=args.max_iter)

            # extract relevant decomposition data form the FlowNMF object
            X = flow_nmf.get_X()
            Z = np.concatenate([Z[np.newaxis, :] for Z in flow_nmf.get_comps_Z().values()], axis=0).squeeze()
            W = np.concatenate([W[np.newaxis, :] for W in flow_nmf.get_comps_W().values()], axis=0).squeeze()
            H_hat = np.concatenate([H[np.newaxis, :] for H in flow_nmf.get_comps_H().values()], axis=0).squeeze()
            NPD = np.concatenate([pZ[np.newaxis, :] for pZ in flow_nmf.get_comps_NPD().values()], axis=0).squeeze()
            WLL = flow_nmf.get_LL_total()

            S = S.numpy()
            R = np.abs(S - X)

            np.save(f'{logdir_full}/X.npy', X)
            np.save(f'{logdir_full}/Z.npy', Z)
            np.save(f'{logdir_full}/W.npy', W)
            np.save(f'{logdir_full}/H.npy', H_hat)
            np.save(f'{logdir_full}/NPD.npy', NPD)
            np.save(f'{logdir_full}/WLL.npy', WLL)
            np.save(f'{logdir_full}/R.npy', R)
            np.save(f'{logdir_full}/S.npy', S)
        else:
            print(f'Loading pre-computed results from {logdir_full}')
            X = np.load(f'{logdir_full}/X.npy')
            Z = np.load(f'{logdir_full}/Z.npy')
            W = np.load(f'{logdir_full}/W.npy')
            H_hat = np.load(f'{logdir_full}/H.npy')
            NPD = np.load(f'{logdir_full}/NPD.npy')
            WLL = np.load(f'{logdir_full}/WLL.npy')
            R = np.load(f'{logdir_full}/R.npy')
            S = np.load(f'{logdir_full}/S.npy')

        if Wn_rescale:
            Wn = np.linalg.norm(W, axis=1)
            H_hat *= Wn

    elif args.method == 'DDSv2' or args.method == 'DDSv3':
        # re-store result if previously computed for this config
        if not os.path.isfile(f'{logdir_full}/X.npy'):
            if args.method == 'DDSv2':
                flow_nmf = FlowNMF2(S, flows_dict, args.n_cps, nll_weight=lw, Z_init=Z_init, H_init=H_init, lr=lr).to(device)
            elif args.method == 'DDSv3':
                flow_nmf = FlowNMF3(S, flow, args.n_cps, nll_weight=lw, Z_init=Z_init, H_init=H_init, lr=lr).to(device)
            flow_nmf.fit(max_iter=args.max_iter)

            # extract relevant decomposition data form the FlowNMF object
            X = flow_nmf.get_X()
            Z = np.concatenate([Z[np.newaxis, :] for Z in flow_nmf.get_comps_Z().values()], axis=0)
            W = np.concatenate([W[np.newaxis, :] for W in flow_nmf.get_comps_W().values()], axis=0)
            H = np.concatenate([H[np.newaxis, :] for H in flow_nmf.get_comps_H().values()], axis=0)
            NPD = np.concatenate([pZ[np.newaxis, :] for pZ in flow_nmf.get_comps_NPD().values()], axis=0)

            S = S.numpy()
            R = np.abs(S - X)

            np.save(f'{logdir_full}/X.npy', X)
            np.save(f'{logdir_full}/Z.npy', Z)
            np.save(f'{logdir_full}/W.npy', W)
            np.save(f'{logdir_full}/H.npy', H)
            np.save(f'{logdir_full}/NPD.npy', NPD)
            np.save(f'{logdir_full}/R.npy', R)
            np.save(f'{logdir_full}/S.npy', S)
        else:
            X = np.load(f'{logdir_full}/X.npy')
            Z = np.load(f'{logdir_full}/Z.npy')
            W = np.load(f'{logdir_full}/W.npy')
            H = np.load(f'{logdir_full}/H.npy')
            NPD = np.load(f'{logdir_full}/NPD.npy')
            R = np.load(f'{logdir_full}/R.npy')
            S = np.load(f'{logdir_full}/S.npy')

        if Wn_rescale:
            Wn = np.linalg.norm(W, axis=1)[:, :, np.newaxis]
            Wn[Wn == 0] = 1 # to avoid potential NaNs gCAR/fCAR computation below, in case there happen to be 0-vectors in W
            H *= Wn

        H_hat = H.sum(axis=1) # [notes, n_cps, time] -> [notes, time]

    # method-agnostic evaluation
    H_y = np.copy(pianoroll[note_range_low:note_range_high+1])
    H_y[H_y > 0] = 1
    H_ysig = np.copy(H_y)
    H_ysig[H_ysig == 0] = -1
    H_yinv = -H_y + 1

    np.save(f'{logdir_full}/H_hat.npy', H_hat)
    np.save(f'{logdir_full}/H_y.npy', H_y)

    gCAR = (H_hat * H_y).sum() / H_hat.sum() # global correct attribution rate
    fCAR = (H_hat * H_y).sum(0) / (H_hat.sum(0) + 1e-300) # frame-wise correct attribution rate
    mfCAR = np.mean(fCAR[~np.isnan(fCAR)]) # mean frame-wise correct attribution rate

    fig = plt.figure(figsize=(20, 1)); plt.plot(fCAR);
    plt.title(f'{args.method} | mean frame-wise CAR {mfCAR:.4f} | global CAR {gCAR:.4f}');
    save_fig_safely(fig, logdir_full, 'car')

    H_score = H_ysig * H_hat

    import matplotlib.colors as colors
    divnorm = colors.TwoSlopeNorm(vmin=H_score.min(), vcenter=0., vmax=H_score.max())
    fig = plt.figure(figsize=(20, 4)); ax = plt.gca(); ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False);
    plt.title(f'{args.method} | {piece} | {mus_piece_factory.instrument}')
    quad_mesh = librosa.display.specshow(H_score, x_axis='frames', y_axis='frames', cmap='PiYG', norm=divnorm);
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cbax = make_axes_locatable(ax).append_axes('left', f'1%', pad=f'5%')
    cb = fig.colorbar(quad_mesh, cax=cbax)
    save_fig_safely(fig, logdir_full, 'H_score')

    thr, F = get_optimal_threshold(H_hat, H_y, n_vals=100, metric='F', incl_metric=True)

    fig = plot_eval(H_hat, H_y, float(hp.sr / hp.hop_length), thr,
                    noterange=(note_range_low, note_range_high+1), figsize=(20, 4), title=args.method);
    save_fig_safely(fig, logdir_full, 'eval_optim_thr')

    fig = plot_reconstruction_performance(R, figsize=(20, 4));
    save_fig_safely(fig, logdir_full, 'R')

    fig = plot_pr_roc_curves(H_hat, H_y, figsize=(20, 4));
    save_fig_safely(fig, logdir_full, 'pr_roc_curves')

    # method-specific additional evaluation
    if args.method == 'NMF':
        pass
    elif args.method == 'DDSv1':
        fig = plot_matrix(NPD, 'Log-Likelihoods [nats per dim] of Z entries, and their H-weighted sum (WLL) as contributing to the cost',
                          curve_right=WLL, label_right='WLL', color_right='tab:orange', alpha_right=1.0,
                          xlabel='time', ylabel='note', figsize=(20, 4));
        save_fig_safely(fig, logdir_full, 'likelihoods')

        fig = plot_matrix(np.log1p(np.linalg.norm(W, axis=1)),
                          'Log-Norms of W vectors (log to deal with outliers) = log(1 + norm(W, axis=frequency))',
                          xlabel='time', ylabel='note', figsize=(20, 4));
        save_fig_safely(fig, logdir_full, 'W_norms_log')

        fig = plot_matrix(np.linalg.norm(Z, axis=1), 'Norms of Z vectors',
                          xlabel='time', ylabel='note', figsize=(20, 4));
        save_fig_safely(fig, logdir_full, 'Z_norms')

    elif args.method == 'DDSv2' or args.method == 'DDSv3':
        fig = plot_matrix(NPD.T, 'Log-Likelihoods [nats per dim] of Z entries',
                          xlabel='note', ylabel='component', figsize=(20, 4));
        save_fig_safely(fig, logdir_full, 'likelihoods')

        fig = plot_matrix(np.log1p(np.linalg.norm(W, axis=1)).T,
                          'Log-Norms of W vectors (log to deal with outliers) = log(1 + norm(W, axis=frequency))',
                          xlabel='note', ylabel='component', figsize=(20, 4));
        save_fig_safely(fig, logdir_full, 'W_norms_log')

        fig = plot_matrix(np.linalg.norm(Z, axis=1).T, 'Norms of Z vectors',
                          xlabel='note', ylabel='component', figsize=(20, 4));
        save_fig_safely(fig, logdir_full, 'Z_norms')


    at_05 = calc_metrics(H_hat, H_y, thr=0.5)
    at_thr = calc_metrics(H_hat, H_y, thr=thr)
    F_at_05, P_at_05, R_at_05 = at_05['F'], at_05['P'], at_05['R']
    F_at_thr, P_at_thr, R_at_thr = at_thr['F'], at_thr['P'], at_thr['R']
    auc_pr, auc_roc = calc_aucs(H_hat, H_y)
    print(f'AC = {gCAR:.5f} | AUC-PR = {auc_pr:.5f} | AUC-ROC = {auc_roc:.5f} | F = {F:.5f}')

    # log quantifiers into CSV logfile
    log_line = f'{args.method} {args.hp} {args.train_set} MAPS {mus_piece_factory.instrument} MUS {args.piece} {piece} M36-95 ' + \
               f'{args.max_iter} {args.max_frames} {args.first_n_sec} {args.n_cps} {args.dds_step_size} ' + \
               f'{gCAR:.5f} {mfCAR:.5f} ' + \
               f'{auc_pr:.5f} {auc_roc:.5f} ' + \
               f'{F_at_05:.5f} {P_at_05:.5f} {R_at_05:.5f} ' + \
               f'{F_at_thr:.5f} {P_at_thr:.5f} {R_at_thr:.5f} {thr:.2f} ' + \
               f'{R.sum(0).mean():.5f}' + (f' {NPD.mean():.5f}' if 'DDS' in args.method else '')

    for f in [open(f'{logdir_full}/grid.csv', 'a'), open(f'logs/{args.csvfile}.csv', 'a'),
              open(f'logs/{args.csvfile}_{args.hp}_{args.first_n_sec:03}.csv', 'a')]:
        f.write(f'{log_line}\n')
        f.close()

if __name__ == '__main__':
    main()
