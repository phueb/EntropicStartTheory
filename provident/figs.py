import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
from typing import List, Tuple, Union

from provident import configs


def human_format(num, pos):  # pos is required for formatting mpl axis ticklabels
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format(num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def make_summary_fig(summaries: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str, int]],
                     ylabel: str,
                     title: str = '',
                     palette_ids: List[int] = None,
                     figsize: Tuple[int, int] = None,
                     ylims: List[float] = None,
                     xlims: List[float] = None,
                     log_x: bool = False,
                     start_x_at_zero: bool = False,
                     y_grid: bool = False,
                     plot_max_line: bool = False,
                     plot_max_lines: bool = False,
                     legend_labels: Union[None, list] = None,
                     vlines: List[int] = None,
                     vline: int = None,
                     legend_loc: str = 'lower right',
                     verbose: bool = False,
                     ):
    # fig
    fig, ax = plt.subplots(figsize=figsize, dpi=configs.Figs.dpi)
    plt.title(title)
    ax.set_xlabel('Training step (mini batch)', fontsize=configs.Figs.axlabel_fs)
    ax.set_ylabel(ylabel, fontsize=configs.Figs.axlabel_fs)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.xaxis.set_major_formatter(FuncFormatter(human_format))
    if y_grid:
        ax.yaxis.grid(True)
    if ylims is not None:
        ax.set_ylim(ylims)
    if log_x:
        ax.set_xscale('log')
    elif start_x_at_zero:
        ax.set_xlim(xmin=0, xmax=summaries[0][0][-1])
    if xlims is not None:
        ax.set_xlim(xlims)

    # palette
    num_summaries = len(summaries)
    palette = np.asarray(sns.color_palette('hls', num_summaries))
    if palette_ids is not None:
        colors = iter(palette[palette_ids])
    else:
        colors = iter(palette)

    if legend_labels is not None:
        legend_labels = iter(legend_labels)

    first_r = True
    first_c = True

    # plot summary
    max_ys = []
    for x, y_mean, h, label, n in summaries:
        max_ys.append(max(y_mean))

        if log_x:
            x += 1  # can't set x-lim to 0 when using log x-axis, so use 1 instead

        if legend_labels is not None:
            try:
                label = next(legend_labels)
            except StopIteration:
                raise ValueError('Not enough values in ALTERNATIVE_LABELS')

        try:
            color = next(colors)
        except StopIteration:
            raise ValueError('Not enough values in PALETTE_IDS')

        if verbose:
            for mean_i, std_i in zip(y_mean, h):
                print(f'mean={mean_i:>6.2f} h={std_i:>6.2f}')

        # if passing multiple summaries, do not label all
        if 'reverse=True' in label:
            color = 'C1'
            if not first_r:
                label = '__nolegend__'
            else:
                label = 'complex first'
                first_r = False
        else:
            color = 'C0'
            if not first_c:
                label = '__nolegend__'
            else:
                label = 'simple first'
                first_c = False

        ax.plot(x, y_mean, '-', linewidth=configs.Figs.lw, color=color,
                label=label, zorder=3 if n == 8 else 2)
        ax.fill_between(x, y_mean + h, y_mean - h, alpha=0.5, color='grey')

    # legend
    if title:
        plt.legend(fontsize=configs.Figs.leg_fs, frameon=False, loc=legend_loc, ncol=1)
    else:
        plt.legend(bbox_to_anchor=(0.5, 1.0),
                   borderaxespad=1.0,
                   fontsize=configs.Figs.leg_fs,
                   frameon=False,
                   loc='lower center',
                   ncol=3,
                   )

    # max line
    if plot_max_line:
        ax.axhline(y=max(max_ys), color='grey', linestyle=':', zorder=1)
    if plot_max_lines:
        for max_y in max_ys:
            ax.axhline(y=max_y, color='grey', linestyle=':', zorder=1)
            print('y max={}'.format(max_y))
    # vertical lines
    if vlines:
        for vline in vlines:
            if vline == 0:
                continue
            print(x[-1], vline / len(vlines))
            ax.axvline(x=x[-1] * (vline / len(vlines)), color='grey', linestyle=':', zorder=1)
    if vline:
        ax.axvline(x=vline, color='grey', linestyle=':', zorder=1)
    plt.tight_layout()
    return fig


def plot_singular_values(ys: List[np.ndarray],
                         max_s: int,
                         fontsize: int = 12,
                         figsize: Tuple[int] = (5, 5),
                         markers: bool = False,
                         label_all_x: bool = True,
                         ):
    fig, ax = plt.subplots(1, figsize=figsize, dpi=None)
    plt.title('SVD of simulated co-occurrence matrix', fontsize=fontsize)
    ax.set_ylabel('Singular value', fontsize=fontsize)
    ax.set_xlabel('Singular Dimension', fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    x = np.arange(max_s) + 1  # num columns
    ax.xaxis.grid(True)
    if label_all_x:
        ax.set_xticks(x)
        ax.set_xticklabels(x)
    # plot
    for n, y in enumerate(ys):
        ax.plot(x, y, label='simulation {}'.format(n + 1), linewidth=2)
        if markers:
            ax.scatter(x, y)
    ax.legend(loc='upper right', frameon=False, fontsize=fontsize)
    plt.tight_layout()
    plt.show()


def plot_heatmap(mat,
                 x_tick_labels: np.ndarray,
                 y_tick_labels: np.ndarray,
                 label_interval: int = 1,
                 dpi: int = 163 * 1,
                 num_cols: int = 512,
                 ):
    fig, ax = plt.subplots(figsize=(12, 12), dpi=dpi)
    plt.title('', fontsize=5)

    # plot subset of columns (to speed computation)
    mat = mat[:, :num_cols]

    # heatmap
    print('Plotting heatmap...')
    ax.imshow(mat,
              # aspect='equal',
              cmap=plt.get_cmap('Greys'),
              interpolation='nearest')

    # x ticks
    x_tick_labels_spaced = []
    for i, l in enumerate(x_tick_labels):
        x_tick_labels_spaced.append(l if i % label_interval == 0 else '')

    num_cols = len(mat.T)
    ax.set_xticks(np.arange(num_cols))
    ax.xaxis.set_ticklabels(x_tick_labels_spaced, rotation=90, fontsize=2)

    # y ticks
    y_tick_labels_spaced = []
    for i, l in enumerate(y_tick_labels):
        y_tick_labels_spaced.append(l if i % label_interval == 0 else '')

    num_rows = len(mat)
    ax.set_yticks(np.arange(num_rows))
    ax.yaxis.set_ticklabels(y_tick_labels_spaced,  # no need to reverse (because no extent is set)
                            rotation=0, fontsize=2)

    # remove spacing between plot and tick labels
    ax.tick_params(axis='both', which='both', pad=-2)

    # remove tick lines
    lines = (ax.xaxis.get_ticklines() +
             ax.yaxis.get_ticklines())
    plt.setp(lines, visible=False)
    plt.show()

