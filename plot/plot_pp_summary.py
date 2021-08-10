from typing import Optional, List, Tuple
from pathlib import Path


from ludwig.results import gen_param_paths

from childesrnnlm import __name__
from childesrnnlm.figs import make_summary_fig
from childesrnnlm.params import param2default, param2requests
from childesrnnlm.summary import make_summary

LUDWIG_DATA_PATH: Optional[Path] = Path('/media/ludwig_data')
RUNS_PATH = None  # configs.Dirs.runs  # configs.Dirs.runs if loading runs locally or None if loading data from ludwig
WHICH_PP = 'train'

LABEL_N: bool = True                       # add information about number of replications to legend
PLOT_MAX_LINES: bool = False                # plot horizontal line at best overall performance
PLOT_MAX_LINE: bool = False                 # plot horizontal line at best performance for each param
PALETTE_IDS: Optional[List[int]] = [1, 0]   # re-assign colors to each line
V_LINES: Optional[List[int]] = None       # add vertical lines to highlight time slices
LABELS: Optional[List[str]] = None  # ['reverse age-ordered', 'age-ordered']  # custom labels for figure legend
FIG_SIZE: Tuple[int, int] = (6, 4)  # in inches
Y_LIMS: List[float] = [100, 10_000]  #  [0.50, 0.60]
Y_LABEL: str = f'Perplexity \n +/- 95%-CI'
CONFIDENCE: float = 0.95
TITLE = ''

# collect summaries
summaries = []
project_name = __name__
for p, label in gen_param_paths(project_name,
                                param2requests,
                                param2default,
                                runs_path=RUNS_PATH,
                                ludwig_data_path=LUDWIG_DATA_PATH,
                                label_n=LABEL_N):
    pattern = f'{WHICH_PP}_pp'
    summary = make_summary(pattern, p, label, CONFIDENCE)  # summary contains: x, mean_y, std_y, label, job_id
    summaries.append(summary)
    print(f'--------------------- End section {p.name}')
    print()

summaries = sort_and_print_summaries(summaries)

# plot
fig = make_summary_fig(summaries,
                       Y_LABEL,
                       log_y=True,
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