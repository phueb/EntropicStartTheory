import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
from typing import List, Tuple, Union

from childesrnnlm import configs


def human_format(num, pos):  # pos is required for formatting mpl axis ticklabels
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format(num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def make_summary_fig(summaries: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str, Union[int, None]]],
                     ylabel: str,
                     title: str = '',
                     palette_ids: List[int] = None,
                     figsize: Tuple[int, int] = None,
                     ylims: List[float] = None,
                     xlims: List[float] = None,
                     log_y: bool = False,
                     start_x_at_zero: bool = False,
                     y_grid: bool = True,
                     plot_max_line: bool = False,
                     plot_max_lines: bool = False,
                     legend_labels: Union[None, list] = None,
                     vlines: List[int] = None,
                     hlines: List[int] = None,
                     vline: int = None,
                     legend_loc: str = 'lower right',
                     annotate: bool = False,
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
    if log_y:
        ax.set_yscale('log')
    if start_x_at_zero:
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
    for x, y_mean, h, label, job_id in summaries:
        max_ys.append(max(y_mean))

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
            print()
            print(label)
            for mean_i, h_i in zip(y_mean, h):
                print(f'mean={mean_i:>6.4f} h={h_i:>6.4f}')

        # add annotations which are helpful to distinguish individual trajectories
        if annotate and 'reverse=False' in label:
            ax.annotate(job_id,
                        xy=(x[-1] + x[-1] * 0.01, y_mean[-1]),
                        xycoords='data',
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=3,
                        # color='C0',
                        )

        # if passing individual trajectories (not average trajectories), do not label all
        if 'shuffle_sentences=True' not in label and len(summaries) == 2 \
                and ('reverse=True' in label
                     or 'start=none' in label
                     or 'yxb' in label
                     or 'yxy' in label
        ):
            color = 'C1'
            if not first_r:
                label = '__nolegend__'
            else:
                first_r = False
        elif 'shuffle_sentences=True' not in label and len(summaries) ==2 \
                and ('reverse=False' in label
                     or 'start=entropic' in label
                     or 'axy' in label
        ):
            color = 'C0'
            if not first_c:
                label = '__nolegend__'
            else:
                first_c = False

        elif 'rxy' in label:
            color = 'C2'

        ax.plot(x, y_mean, '-',
                linewidth=configs.Figs.lw,
                color=color,
                label=label,
                )
        ax.fill_between(x,
                        y_mean + h,
                        y_mean - h,
                        alpha=0.2,
                        color=color)

    # legend
    if title:
        plt.legend(fontsize=configs.Figs.leg_fs, frameon=False, loc=legend_loc, ncol=1)
    else:
        plt.legend(bbox_to_anchor=(0.5, 1.0),
                   borderaxespad=1.0,
                   fontsize=configs.Figs.leg_fs,
                   frameon=False,
                   loc='lower center',
                   ncol=4,
                   )

    # max line
    if plot_max_line:
        ax.axhline(y=max(max_ys), color='grey', linestyle=':', zorder=1)
    if plot_max_lines:
        for max_y in max_ys:
            ax.axhline(y=max_y, color='grey', lw=1, linestyle='-', zorder=1)
            print('y max={}'.format(max_y))

    # horizontal lines
    if hlines:
        for hline in hlines:
            if hline == 0:
                continue
            ax.axhline(y=hline, color='grey', linestyle=':', zorder=1)

    # vertical lines
    if vlines:
        for vline in vlines:
            if vline == 0:
                continue
            ax.axvline(x=vline, color='grey', linestyle=':', zorder=1)
    if vline:
        ax.axvline(x=vline, color='grey', linestyle=':', zorder=1)

    plt.tight_layout()
    return fig


def get_y_label_and_lims(performance_name: str,
                         direction: str,
                         location: str,
                         context_type: str,
                         add_confidence_interval_to_label: bool,
                         ) -> Tuple[str, List[float]]:
    if performance_name == 'ba':
        y_label = 'Balanced Accuracy'
        y_lims = [0.5, 0.7]
    elif performance_name == 'si':
        y_label = 'Silhouette Score'
        y_lims = None
    elif performance_name == 'sd':
        y_label = 'S_Dbw Score'
        y_lims = None
    elif performance_name == 'ma':
        y_label = 'Magnitude of Representations'
        y_lims = None
    elif performance_name == 'pr1':
        y_label = 'Divergence actual vs. theoretical prototype'
        y_lims = None
    elif performance_name == 'pr2':
        y_label = 'Divergence exemplars vs theoretical prototype'
        y_lims = None
    elif performance_name == 'pd':
        y_label = 'Pairwise Divergence'
        y_lims = None
    elif performance_name == 'pe':
        y_label = 'Pairwise Euclidean Distance'
        y_lims = None
    elif performance_name == 'cs':
        y_label = 'Cosine Similarity'
        y_lims = None
    elif performance_name == 'cc':
        y_label = 'Cosine Similarity'
        y_lims = None
    elif performance_name == 'op':
        y_label = 'Divergence origin vs. theoretical prototype'
        y_lims = None
    elif performance_name == 'en':
        y_label = 'Entropy of Representations'
        y_lims = None
    elif performance_name == 'eo':
        y_label = 'Entropy at Origin'
        y_lims = None
    elif performance_name == 'fr':
        y_label = 'Fragmentation'
        y_lims = [0.7, 0.9] if location == 'out' else None
    elif performance_name == 'cd':
        y_label = 'Within-probe Context Divergence'
        y_lims = [0, 0.6]
    elif performance_name == 'ds':
        y_label = ' Divergence from superordinate'
        y_lims = [0, 0.7]
    else:
        raise AttributeError

    if direction == 'l':
        y_label += '\nLeft Context Words'
    elif direction == 'c':
        y_label += '\nProbe Words'
    elif direction == 'r':
        y_label += '\nRight Context Words'
    else:
        raise AttributeError('Invalid arg to direction')

    if location == 'inp':
        y_label += '\nInput Layer'
    elif location == 'out':
        y_label += '\nOutput Layer'
    else:
        raise AttributeError('Invalid arg to location')

    if context_type == 'n':
        y_label += '\nNon-contextualized'
    elif context_type == 'o':
        y_label += '\nContextualized'
    elif context_type == 'm':
        if direction == 'l':
            y_label += '\nLeft Context Word without Left Context'
        elif direction == 'c':
            y_label += '\nContext without Probe'
        elif direction == 'r':
            y_label += '\nRight Context Word without Right Context'
    else:
        raise AttributeError('Invalid arg to context_type')

    if add_confidence_interval_to_label:
        y_label += '\n+/- 95%-CI'

    return y_label, y_lims
