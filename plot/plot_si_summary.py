from typing import Optional, List, Tuple
from pathlib import Path

from ludwig.results import gen_param_paths

from entropicstarttheory import __name__
from entropicstarttheory.figs import make_summary_fig
from entropicstarttheory.summary import make_summary
from entropicstarttheory.params import param2default, param2requests

LUDWIG_DATA_PATH: Optional[Path] = Path('/media/ludwig_data')
RUNS_PATH = None  # config.Dirs.runs  # config.Dirs.runs if using local plot or None if using plot form Ludwig
SI_TYPE: str = 'si_n'
PROBES_NAME: str = 'sem-2021'

LABEL_N: bool = True                       # add information about number of replications to legend
PLOT_MAX_LINE: bool = False                 # plot horizontal line at best performance for each param
PLOT_MAX_LINES: bool = False                # plot horizontal line at best overall performance
PALETTE_IDS: Optional[List[int]] = None   # re-assign colors to each line
V_LINES: Optional[List[int]] = None       # add vertical lines to highlight time slices
LABELS: Optional[List[str]] = None  # ['reverse age-ordered', 'age-ordered']  # custom labels for figure legend
FIG_SIZE: Tuple[int, int] = (6, 4)  # in inches
Y_LIMS: List[float] = None  # [-0.06, 0.0]
CONFIDENCE: float = 0.95
TITLE = ''  # f'{SI_NAME}_{PROBES_NAME}.csv'

if SI_TYPE == 'si_n':
    Y_LABEL: str = f'Silhouette Score at Input\n +/- 95%-CI'
elif SI_TYPE == 'si_o':
    Y_LABEL: str = f'Silhouette Score at Hidden\n +/- 95%-CI'
else:
    raise AttributeError('Invalid SI_TYPE')

# collect summaries
summaries = []
project_name = __name__
for p, label in gen_param_paths(project_name,
                                param2requests,
                                param2default,
                                runs_path=RUNS_PATH,
                                ludwig_data_path=LUDWIG_DATA_PATH,
                                label_n=LABEL_N):
    pattern = f'{SI_TYPE}_{PROBES_NAME}'
    summary = make_summary(pattern, p, label, CONFIDENCE)  # summary contains: x, mean_y, std_y, label, n
    summaries.append(summary)
    print(f'--------------------- End section {p.name}')
    print()

# sort data
summaries = sorted(summaries, key=lambda s: s[1][-1], reverse=True)
if not summaries:
    raise SystemExit('No data found')

# print to console
for s in summaries:
    _, y_mean, y_std, label, n = s
    print(label)
    print(y_mean)
    print(y_std)
    print()

# plot
fig = make_summary_fig(summaries,
                       Y_LABEL,
                       title=TITLE,
                       palette_ids=PALETTE_IDS,
                       figsize=FIG_SIZE,
                       ylims=Y_LIMS,
                       legend_labels=LABELS,
                       vlines=V_LINES,
                       plot_max_lines=PLOT_MAX_LINES,
                       plot_max_line=PLOT_MAX_LINE,
                       legend_loc='best',
                       )
fig.show()