from typing import Optional, List, Tuple
from pathlib import Path


from ludwig.results import gen_param_paths

from childesrnnlm import __name__
from childesrnnlm.figs import make_summary_fig
from childesrnnlm.summary import make_summary, sort_and_print_summaries
from childesrnnlm.params import param2default, param2requests

LUDWIG_DATA_PATH: Optional[Path] = Path('/media/ludwig_data')
RUNS_PATH = None  # config.Dirs.runs if using local plot or None if using plot form Ludwig
BA_TYPE: str = 'ba_n'
PROBES_NAME: str = 'sem-2021'

LABEL_N: bool = True                       # add information about number of replications to legend
PLOT_MAX_LINE: bool = False                 # plot horizontal line at best performance for each param
PLOT_MAX_LINES: bool = True                # plot horizontal line at best overall performance
PALETTE_IDS: Optional[List[int]] = None   # re-assign colors to each line
V_LINES: Optional[List[int]] = None       # add vertical lines to highlight time slices
LABELS: Optional[List[str]] = None  # ['reverse age-ordered', 'age-ordered']  # custom labels for figure legend
FIG_SIZE: Tuple[int, int] = (6, 4)  # in inches
Y_LIMS: List[float] = [0.50, 0.70]
CONFIDENCE: float = 0.95
TITLE = ''  # f'{BA_TYPE}_{PROBES_NAME}.csv'

if BA_TYPE == 'ba_n':
    Y_LABEL: str = f'Balanced Accuracy at Input\n +/- 95%-CI'
elif BA_TYPE == 'ba_o':
    Y_LABEL: str = f'Balanced Accuracy at Hidden\n +/- 95%-CI'
else:
    raise AttributeError('Invalid BA_TYPE')

# collect summaries
summaries = []
project_name = __name__
for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         runs_path=RUNS_PATH,
                                         ludwig_data_path=LUDWIG_DATA_PATH,
                                         label_n=LABEL_N):

    pattern = f'{BA_TYPE}_{PROBES_NAME}'
    summary = make_summary(pattern, param_path, label, CONFIDENCE)
    summaries.append(summary)  # summary contains: x, mean_y, std_y, label, job_id
    print(f'--------------------- End section {param_path.name}')
    print()

summaries = sort_and_print_summaries(summaries)

# plot
fig = make_summary_fig(summaries,
                       ylabel=Y_LABEL,
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
