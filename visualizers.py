import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import mir_eval
import mir_eval.display

from metrics import thresh, calc_metrics


def save_fig_safely(fig, logdir, name, ext='png'):
    try:
        fig.savefig(f'{logdir}/{name}.{ext}', bbox_inches='tight')
        plt.close(fig)
    except:
        pass


def plot_perf_curve(x_metric, y_metric, thresholds=None, ax=None,
                    label='', xlabel='', ylabel='', title=''):
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()

    ax.plot(x_metric, y_metric, label=label);
    ax.set_title(title);
    ax.set_xlabel(xlabel);
    ax.set_ylabel(ylabel);

    if thresholds is not None:
        # add re-scaled thresholds values on secondary axis
        thr_max = thresholds.max()
        ax_r = ax.secondary_yaxis('right', functions=(lambda x: x * thr_max, lambda x: x / thr_max))
        ax_r.set_ylabel('Thresholds')
        ax.plot(x_metric[:len(thresholds)], thresholds / thr_max, linestyle='--', label='thresholds')

    ax.legend(loc='upper right');


def plot_pr_roc_curves(activations, labels, figsize=(14, 4)):
    '''Produces a single figure plotting both PR and ROC curves.'''

    from sklearn.metrics import precision_recall_curve, roc_curve, auc

    fig = plt.figure(figsize=figsize)
    ax_pr = plt.subplot(1, 2, 1)
    ax_roc = plt.subplot(1, 2, 2)

    precision, recall, thresholds = precision_recall_curve(labels.flatten(), activations.flatten())
    auc_pr = auc(recall, precision)

    plot_perf_curve(recall, precision, thresholds, ax=ax_pr,
                    label='PR curve', xlabel='Recall', ylabel='Precision',
                    title=f'Precision-Recall (PR) curve\nAUC-PR = {auc_pr:.5f}')

    fpr, tpr, thresholds = roc_curve(labels.flatten(), activations.flatten())
    auc_roc = auc(fpr, tpr)

    plot_perf_curve(fpr, tpr, thresholds, ax=ax_roc,
                    label='ROC-curve', xlabel='False Positive Rate', ylabel='True Positive Rate',
                    title=f'Receiver Operating Characteristic (ROC) curve\nAUC-ROC = {auc_roc:.5f}')

    return fig

def get_figsize(array, factor=128):
    assert array.ndim == 2
    return (int(np.round(array.shape[1] / factor)), int(np.round(array.shape[0] / factor)))


def get_range_with_margins(data, margin_fac=1/10, margin_def=1):
    data_min = data.min()
    data_max = data.max()
    margin = ((data_max - data_min) * margin_fac) or margin_def
    data_min -= margin
    data_max += margin
    return data_min, data_max


def plot_matrix(array, title, xlabel='', ylabel='', cbar=True,
                curve_left=None, label_left='', color_left='tab:blue', alpha_left=0.5,
                curve_right=None, label_right='', color_right='tab:orange', alpha_right=0.5,
                cmap='Greys', figsize=None, fac=None, fig_height=0, fig_width=20):

    factor = fac or (array.shape[0]/fig_height if fig_height else array.shape[1]/fig_width)
    figsize = figsize or get_figsize(array, factor=factor)
    fig = plt.figure(figsize=figsize)
    plt.title(title)
    quad_mesh = librosa.display.specshow(
        array, x_axis='frames', y_axis='frames', cmap=cmap, vmin=min(array.min(), 0), vmax=array.max())

    # set axis labels
    ax = plt.gca()
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    if curve_left is not None:
        ax.get_yaxis().set_visible(False) # remove left axis ticks and ticklabels

        cl_min, cl_max = get_range_with_margins(curve_left)

        bins2cl = lambda x: x / array.shape[0] * (cl_max - cl_min) + cl_min
        cl2bins = lambda x: (x - cl_min) / (cl_max - cl_min) * array.shape[0]

        ax_l = ax.secondary_yaxis('left', functions=(bins2cl, cl2bins))
        ax_l.set_ylabel(label_left)

        plt.plot(cl2bins(curve_left), alpha=alpha_left, label=label_left, color=color_left)

    if curve_right is not None:
        cr_min, cr_max = get_range_with_margins(curve_right)

        bins2cr = lambda x: x / array.shape[0] * (cr_max - cr_min) + cr_min
        cr2bins = lambda x: (x - cr_min) / (cr_max - cr_min) * array.shape[0]

        ax_r = ax.secondary_yaxis('right', functions=(bins2cr, cr2bins))
        ax_r.set_ylabel(label_right)

        if curve_right.ndim > 1 and type(label_right) == list:
            lineObjects = plt.plot(cr2bins(curve_right), alpha=alpha_right)
            plt.legend(lineObjects, label_right, loc='upper right')
        else:
            plt.plot(cr2bins(curve_right), alpha=alpha_right, label=label_right, color=color_right)

    if (label_left or label_right) and type(label_left) != list and type(label_right) != list:
        plt.legend(loc='upper right')

    # colorbar
    if cbar:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        cbax = make_axes_locatable(ax).append_axes('right', f'{20/figsize[0]}%', pad=f'{20/figsize[0]*5}%')
        cb = fig.colorbar(quad_mesh, cax=cbax)

    return fig


def plot_reconstruction_performance(R, likelihoods=None, cmap='Greys', figsize=None, fac=None, fig_height=0, fig_width=20):
    '''
    Plot reconstruction performance of NMF / FlowNMF (based on whether likelihoods are provided or not).

    Figsize factor argument `fac` stands for "how many values are plotted per unit of figure size"
    and applies universally for both dimensions of the plot.

    Behavior of figure size specifiers:
     - `fig_height` if non-zero, overrides `fig_width`.
     - `fac` is specified overrides both `fig_height` and `fig_width`.
     - `figsize` if specified overrides all other options.
    '''
    factor = fac or (R.shape[0]/fig_height if fig_height else R.shape[1]/fig_width)
    figsize = figsize or get_figsize(R, factor=factor)
    #figsize = (figsize[0] * 10, figsize[1])
    fig = plt.figure(figsize=figsize)
    plt.title(f'Residuals $R = \| S - \hatS \| $ with total reconstruction error $ \sum R = ${R.sum():.2f}')

    quad_mesh = librosa.display.specshow(R, x_axis='frames', y_axis='frames', cmap=cmap, vmin=0, vmax=R.max())

    # remove axis labels
    ax = plt.gca()
    ax.set_xlabel('samples'); ax.set_ylabel('')
    ax.get_yaxis().set_visible(False)

    # draw reconstruction error curve over the plot with left axis as label
    rec_errors = R.sum(axis=0)
    _, re_max = get_range_with_margins(rec_errors)
    bins2re = lambda x: x / R.shape[0] * re_max
    re2bins = lambda x: x / re_max * R.shape[0]
    ax_re = ax.secondary_yaxis('left', functions=(bins2re, re2bins)) #ax_re.set_ylim([0, rec_errors.max()]) # didn't work
    ax_re.set_ylabel('err')
    plt.plot(re2bins(rec_errors), alpha=0.5, label='rec_error')

    # draw sample likelihood curve over the plot with right axis as label
    if likelihoods is not None:
        ll_min, ll_max = get_range_with_margins(likelihoods)
        bins2ll = lambda x: x / R.shape[0] * (ll_max - ll_min) + ll_min
        ll2bins = lambda x: (x - ll_min) / (ll_max - ll_min) * R.shape[0]
        ax_ll = ax.secondary_yaxis('right', functions=(bins2ll, ll2bins))
        ax_ll.set_ylabel('nats/dim')
        plt.plot(ll2bins(likelihoods), alpha=0.5, label='nats/dim')

    plt.legend(loc='upper right')

    # colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cbax = make_axes_locatable(ax).append_axes('right', f'{20/figsize[0]}%', pad=f'{20/figsize[0]*5}%')
    cb = fig.colorbar(quad_mesh, cax=cbax)

    return fig


def plot_attribution_map(H_score, title='', cbar=False, figsize=(20, 4), fac=None, fig_height=0, fig_width=20):
    import matplotlib.colors as colors
    divnorm = colors.TwoSlopeNorm(vmin=H_score.min(), vcenter=0., vmax=H_score.max())

    fig = plt.figure(figsize=figsize)
    plt.title(title)

    quad_mesh = librosa.display.specshow(H_score, x_axis='frames', y_axis='frames', cmap='PiYG', norm=divnorm);

    # remove axis labels
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # colorbar
    if cbar:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        cbax = make_axes_locatable(ax).append_axes('left', f'1%', pad=f'5%')
        cb = fig.colorbar(quad_mesh, cax=cbax)

    return fig


# From https://github.com/lmartak/amt-wavenet/blob/master/utils/renderers.py
def piano_roll2midi_events(piano_roll, fs, noterange=(21, 109), dynamic=False):
    '''Converts piano roll into list of midi events,
    which is a standard format for evaluation in MIR
    required also by `mir_eval.display.piano_roll`.
    '''

    times = np.empty([0,2])
    midis = np.empty([0])
    velocities = list()
    t_start = -1

    for note in np.unique(np.nonzero(piano_roll)[0]):
        for t in range(piano_roll.shape[1]):
            if (t_start == -1 and
                piano_roll[note][t] != 0 and
                t < piano_roll.shape[1]-1):
                t_start = t
            elif (t_start != -1 and
                  (piano_roll[note][t] == 0 or t == piano_roll.shape[1]-1)):
                times = np.append(times, [[t_start/fs, t/fs]], axis=0)
                midis = np.append(midis, [note + noterange[0]], axis=0)
                if dynamic:
                    velocities.append(np.mean(piano_roll[note][t_start:t]))
                t_start = -1

    return (times, midis, velocities) if dynamic else (times, midis)


# From https://github.com/lmartak/amt-wavenet/blob/master/utils/renderers.py
def plot_eval(predictions, labels, fs, thr,
              noterange=(21, 109), retain=True, legend=True, ticker_base=12.0,
              figsize=(4*4, 1*4), title='', savepath=None, ax=None):
    '''Plots IR-style evaluation matrix by comparing
    thresholded labels with thresholded estimations.
    IR task Outcomes are visualized in colors.
    '''

    import matplotlib

    # Obtain thresholded labels and predictions
    p = thresh(predictions, thr, retain).astype(bool)
    l = thresh(labels, thr, retain).astype(bool)

    # true positives (1 AND 1)
    TP_times, TP_midis = piano_roll2midi_events(p * l, fs, noterange)
    # false negatives ((0 XOR 1) AND 1)
    FN_times, FN_midis = piano_roll2midi_events((p ^ l) * l, fs, noterange)
    # false positives ((1 XOR 0) AND 1)
    FP_times, FP_midis = piano_roll2midi_events((p ^ l) * p, fs, noterange)

    if ax is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = ax.get_figure()

    mir_eval.display.piano_roll(TP_times,
                                midi=TP_midis,
                                label='TP',
                                facecolor=(0, 1, 0, 1),
                                linewidth=0,
                                ax=ax)
    mir_eval.display.piano_roll(FN_times,
                                midi=FN_midis,
                                label='FN',
                                facecolor=(1, 0, 0, 1),
                                linewidth=0,
                                ax=ax)
    mir_eval.display.piano_roll(FP_times,
                                midi=FP_midis,
                                label='FP',
                                facecolor=(0, 0, 1, 1),
                                linewidth=0,
                                ax=ax)

    if ax is None:
        ax = fig.get_axes()[0]

    ax.grid(True, which='major')
    ax.grid(True, which='minor', alpha=0.25)
    if noterange:
        ax.set_ybound(lower=noterange[0], upper=noterange[1])
    if legend:
        ax.legend(mode='best')

    metrics = calc_metrics(predictions, labels, thr=thr)
    title_metrics = f'F1 = {metrics["F"]:.2f} | P = {metrics["P"]:.2f} | R = {metrics["R"]:.2f} | thr = {thr:.2f}'
    ax.set_title(f'{title} | {title_metrics}' if title else title_metrics)

    mir_eval.display.ticker_notes(ax=ax)
    loc = matplotlib.ticker.MultipleLocator(base=ticker_base)
    ax.yaxis.set_major_locator(loc)

    ax.tick_params(labelbottom=False, labelleft=True)

    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig

def plot_frames(frames, bs, hist=10, xlabels=[], savepath=None, verbose=True):
    if verbose:
        print(f'min {frames.min().item():.2f}, avg {frames.mean().item():.2f}, ' +
              f'max {frames.max().item():.2f}, std {frames.std().item():.2f}')

    fig, axs = plt.subplots(1, bs + hist, sharey=True, figsize=(bs//2 + hist//2, 4))
    fig.subplots_adjust(wspace=0) # width space between axes

    ys = np.arange(frames.shape[1])
    for i in range(bs):
        axs[i].plot(frames[i].cpu().numpy(), ys)
        axs[i].vlines(0, 0, len(ys), colors='r', alpha=0.5)
        if len(xlabels) == bs:
            axs[i].get_xaxis().set_ticks([]); axs[i].get_xaxis().set_ticklabels([]);
            axs[i].set(xlabel=xlabels[i])
        else:
            axs[i].get_xaxis().set_visible(False)


    if hist:
        gs = axs[0].get_gridspec()
        for ax in axs[bs:]:
            ax.remove()
        ax_hist = fig.add_subplot(gs[bs+1:])
        ax_hist.hist(frames.view(-1).cpu().numpy(), bins=100, log=True)

    return fig
