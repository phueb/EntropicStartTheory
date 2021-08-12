import numpy as np
from typing import Optional
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist, squareform
from scipy.stats import sem, t

from ludwig.results import gen_param_paths


from childesrnnlm import __name__
from childesrnnlm.params import param2default, param2requests
from childesrnnlm.figs import make_summary_fig

LUDWIG_DATA_PATH: Optional[Path] = Path('/media/ludwig_data')
RUNS_PATH: Optional[Path] = None  # configs.Dirs.runs if loading runs locally or None if from ludwig

METRIC = ['cosine', 'euclidean'][0]  # cosine
METHOD = ['single', 'average', 'complete', 'ward'][1]  # complete

WHERE = ['input', 'output'][1]
FIRST_NUM_STEPS = 100  # the number of first training steps (otherwise, script takes long)

PLOT_DENDROGRAM = False
Y_LIMS = [0, 2]
SHOW_NUM_CLUSTERS = 128
LEAF_FONT_SIZE = 6

if PLOT_DENDROGRAM:
    plt.ion()

# init collection of cophenetic coefficients (a matrix per param_name, with shape [num reps, num steps])
param_name2c_mat = {}

project_name = __name__
summaries = []
steps = set()
for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         runs_path=RUNS_PATH,
                                         ludwig_data_path=LUDWIG_DATA_PATH,
                                         label_n=True):

    job_names = [path.stem for path in param_path.glob('*-*-*:*:*_*')]
    print(f'Found job names:')
    print(job_names)

    for row_id, job_name in enumerate(job_names):

        pattern = 'probe_reps_*.npz'
        npz_paths = list(sorted((param_path / job_name / 'saves').glob(pattern)))
        if not npz_paths:
            raise FileNotFoundError(f'Did not find {pattern} in {param_path}')

        # truncate steps to shorten waiting time
        num_steps = min(len(npz_paths), FIRST_NUM_STEPS)
        npz_paths = npz_paths[:num_steps]

        # set up figure for animation
        # note: if plot does not show up, turn off Python Scientific in Pycharm Settings
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

        # load representations at each step
        for col_id, npz_path in enumerate(npz_paths):
            print(f'Loading representations from archive at {npz_path.relative_to(param_path.parent)}')
            with np.load(npz_path, allow_pickle=True) as loaded:
                if WHERE == 'input':
                    probe_reps = loaded['probe_reps_inp']
                else:
                    probe_reps = loaded['probe_reps_out']

            step = int(npz_path.stem[-12:])
            steps.add(step)
            print(f'step={step}')

            # do clustering
            Z = linkage(probe_reps, method=METHOD, metric=METRIC)
            # compute cophenetic coefficient - used to check that linkage is good and if data is nested
            c, cophenetic_sim_mat = cophenet(Z, pdist(probe_reps, metric=METRIC))
            print(f'cophenetic coefficient={c:.2f}')

            # collect cophenetic coefficient
            param_name2c_mat.setdefault(param_path.name, np.zeros((len(job_names), len(npz_paths))))
            param_name2c_mat[param_path.name][row_id][col_id] = c
            print(param_name2c_mat[param_path.name].round(2))

            # plot
            if PLOT_DENDROGRAM:
                plt.title(f'Hierarchical Clustering Dendrogram (truncated)'
                          f'\nprobe representations at {WHERE}'
                          f'\n{label}'  # TODO remove string "n=..."
                          f'\nlinkage method={METHOD}'
                          f'\nstep={step:,}'
                          f'\ncophenetic coeff.={c:.2f}',
                          fontdict={'fontsize': 6})
                ax.set_xlabel('')
                ax.set_ylabel(f'{METRIC} distance')
                d = dendrogram(
                    Z,
                    truncate_mode='lastp',  # show only the last p merged clusters
                    p=SHOW_NUM_CLUSTERS,  # show only the last p merged clusters
                    show_leaf_counts=False,  # otherwise numbers in brackets are counts
                    leaf_rotation=90.,
                    leaf_font_size=LEAF_FONT_SIZE,
                    show_contracted=True,  # to get a distribution impression in truncated branches
                    ax=ax,
                    color_threshold=0,  # do not auto-color clusters if set to zero
                )
                ax.set_ylim(Y_LIMS)  # must be called after dendrogram() call
                ax.set_xticklabels([])
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.tick_params(axis='both', which='both', top=False, right=False)

                # interactive update of the same figure
                fig.tight_layout()  # to fit the title
                fig.canvas.draw()
                fig.canvas.flush_events()
                # remove previous dendrogram
                ax.cla()

    # collect summary for each param_name
    job_id = None
    x = np.array(list(sorted(list(steps))))
    y_mean = param_name2c_mat[param_path.name].mean(axis=0)
    y_sem = sem(param_name2c_mat[param_path.name], axis=0)
    confidence = 0.95
    n = len(job_names)
    h = y_sem * t.ppf((1 + confidence) / 2, n - 1)  # margin of error
    summary = (x, y_mean, h, label, job_id)
    summaries.append(summary)  # summary contains: x, mean_y, margin-of-error, label, job_id

if WHERE == 'output':
    y_lims = [0.4, 0.8]
else:
    y_lims = [0, 1]


# plot cophenetic coefficient across training
# note: this graph should be shown in Pycharm SciView, and will not be drawn if Python Scientific is off
if not PLOT_DENDROGRAM:
    print('Make sure Python Scientific is turned on in Pycharm Settings')
y_label = f'Cophenetic Coefficient' \
          f'\nProbe Representations at {WHERE}' \
          f'\n+/- 95%-CI'
fig = make_summary_fig(summaries,
                       ylabel=y_label,
                       figsize=(6, 4),
                       ylims=y_lims,
                       )
fig.show()
