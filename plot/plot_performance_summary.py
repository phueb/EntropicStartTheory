from typing import Optional, List, Tuple
from pathlib import Path

from ludwig.results import gen_param_paths

from childesrnnlm import __name__
from childesrnnlm.figs import make_summary_fig, get_y_label_and_lims
from childesrnnlm.summary import make_summary, sort_and_print_summaries, save_summary_to_txt
from childesrnnlm.params import param2default, param2requests

LUDWIG_DATA_PATH: Optional[Path] = Path('/media/ludwig_data')
RUNS_PATH = None  # configs.Dirs.runs if loading runs locally or None if loading data from ludwig
PROBES_NAME: str = 'sem-2021'

LABEL_N: bool = True                        # add information about number of replications to legend
PLOT_MAX_LINE: bool = False                 # plot horizontal line at best performance for each param
PLOT_MAX_LINES: bool = False                # plot horizontal line at best overall performance
PALETTE_IDS: Optional[List[int]] = None   # re-assign colors to each line
V_LINES: Optional[List[int]] = None       # add vertical lines to highlight time slices
LABELS: Optional[List[str]] = None  # ['reverse age-ordered', 'age-ordered']  # custom labels for figure legend
FIG_SIZE: Tuple[int, int] = (6, 4)  # in inches
CONFIDENCE: float = 0.95
TITLE = ''
CONTEXT_TYPE = ['n', 'o'][0]
PERFORMANCE_NAME = ['ma',  # 0
                    'ra',  # 1  # TODO test this
                    'ba',  # 2
                    'th',  # 3
                    'dp',  # 4
                    'du',  # 5
                    'ws',  # 6
                    'as',  # 7
                    'ed',  # 8
                    'cs',  # 9
                    'si',  # 10
                    'sd',  # 11
                    'pi',  # 12
                    'ep',  # 13
                    'eo',  # 14
                    'db',  # 15
                    'fi',  # 16
                    'fo',  # 17
                    'co',  # 18
                    'cc',  # 19
                    ][1]

param2requests = {'reverse': [True, False]}

# collect summaries
summaries = []
project_name = __name__
for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         runs_path=RUNS_PATH,
                                         ludwig_data_path=LUDWIG_DATA_PATH,
                                         label_n=LABEL_N):

    pattern = f'{PERFORMANCE_NAME}_{CONTEXT_TYPE}_{PROBES_NAME}'
    summary = make_summary(pattern, param_path, label, CONFIDENCE)
    summaries.append(summary)  # summary contains: x, mean_y, margin-of-error, label, job_id

    save_summary_to_txt(summary, pattern)

    print(f'--------------------- End section {param_path.name}')
    print()

summaries = sort_and_print_summaries(summaries)

# plot
y_label, y_lims = get_y_label_and_lims(PERFORMANCE_NAME,
                                       CONTEXT_TYPE,
                                       add_confidence_interval_to_label=True)
fig = make_summary_fig(summaries,
                       ylabel=y_label,
                       title=TITLE,
                       palette_ids=PALETTE_IDS,
                       figsize=FIG_SIZE,
                       ylims=y_lims,
                       legend_labels=LABELS,
                       vlines=V_LINES,
                       plot_max_lines=PLOT_MAX_LINES,
                       plot_max_line=PLOT_MAX_LINE,
                       )
fig.show()
