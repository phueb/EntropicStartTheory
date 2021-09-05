from typing import Optional, List, Tuple
from pathlib import Path

from ludwig.results import gen_param_paths

from childesrnnlm import __name__
from childesrnnlm.figs import make_summary_fig, get_y_label_and_lims
from childesrnnlm.summary import make_summary, sort_and_print_summaries, save_summary_to_txt
from childesrnnlm.params import param2default, param2requests

LUDWIG_DATA_PATH: Optional[Path] = Path('/media/ludwig_data')
RUNS_PATH = None  # configs.Dirs.runs if loading runs locally or None if loading data from ludwig

LABEL_N: bool = True                        # add information about number of replications to legend
PLOT_MAX_LINE: bool = False                 # plot horizontal line at best performance for each param
PLOT_MAX_LINES: bool = False                # plot horizontal line at best overall performance
PALETTE_IDS: Optional[List[int]] = None   # re-assign colors to each line
V_LINES: Optional[List[int]] = None       # add vertical lines to highlight time slices
LABELS: Optional[List[str]] = None  # ['reverse age-ordered', 'age-ordered']  # custom labels for figure legend
FIG_SIZE: Tuple[int, int] = (6, 4)  # in inches
CONFIDENCE: float = 0.95
TITLE = ''

STRUCTURE_NAME: str = 'sem-2021'

# CAREFUL: using DIRECTION='r' is not useful for analyzing fragmentation of mapping between probes and right contexts,
# because this is measured by fragmentation of probe words.
# (probe words predict right-contexts, but right-contexts don't predict probe words)

DIRECTION = ['l',  # left-of-probe,
             'c',  # center (probe)
             'r',  # right-of-probe:
             ][2]
LOCATION = ['inp',  # input layer
            'out',  # output layer
            ][1]

CONTEXT_TYPE = ['n',  # no context + probe
                'o',  # ordered context + probe
                'm',  # "minus 1" - this means ordered context up to probe (excluding probe)
                ][1]

PERFORMANCE_NAME = ['ba',  # 0
                    'si',  # 1
                    'sd',  # 2
                    'ma',  # 3
                    'pr1',  # 4  defined for LOCATION='inp' only
                    'pr2',  # 5  defined for LOCATION='out' only
                    'pd',  # 6   defined for LOCATION='out' only
                    'pe',  # 7   defined for LOCATION='inp' only
                    'cs',  # 8
                    'cc',  # 9
                    'op',  # 10
                    'en',  # 11
                    'eo',  # 12
                    'fr',  # 13
                    ][13]

pattern = f'{PERFORMANCE_NAME}_{STRUCTURE_NAME}_{DIRECTION}_{LOCATION}_{CONTEXT_TYPE}'


# param2requests = {'reverse': [True, False]}

# collect summaries
summaries = []
project_name = __name__
for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         runs_path=RUNS_PATH,
                                         ludwig_data_path=LUDWIG_DATA_PATH,
                                         label_n=LABEL_N):

    summary = make_summary(pattern, param_path, label, CONFIDENCE)
    summaries.append(summary)  # summary contains: x, mean_y, margin-of-error, label, job_id

    save_summary_to_txt(summary, pattern)

    print(f'--------------------- End section {param_path.name}')
    print()

summaries = sort_and_print_summaries(summaries)

# plot
y_label, y_lims = get_y_label_and_lims(PERFORMANCE_NAME,
                                       DIRECTION,
                                       LOCATION,
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
