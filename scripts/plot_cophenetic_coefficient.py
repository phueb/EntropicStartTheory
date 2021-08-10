import numpy as np
from typing import Optional
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from scipy.stats import sem, t

from ludwig.results import gen_param_paths

from childesrnnlm import __name__
from childesrnnlm.params import param2default, param2requests
from childesrnnlm.figs import make_summary_fig

LUDWIG_DATA_PATH: Optional[Path] = Path('/media/ludwig_data')
RUNS_PATH: Optional[Path] = None  # configs.Dirs.runs if loading runs locally or None if from ludwig

METRIC = 'cosine'
METHOD = 'single'

FIRST_NUM_STEPS = 3  # TODO debugging

PLOT_DENDROGRAM = False
Y_LIMS = [0, 1]
SHOW_NUM_CLUSTERS = 128
LEAF_FONT_SIZE = 6

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
        npz_paths = list(sorted((param_path / job_name / 'saves').glob(pattern)))[:FIRST_NUM_STEPS]
        if not npz_paths:
            raise FileNotFoundError(f'Did not find {pattern} in {param_path}')

        # load representations at each step
        for col_id, npz_path in enumerate(npz_paths):

            print(f'Loading representations from archive at {npz_path.relative_to(param_path.parent)}')
            with np.load(npz_path, allow_pickle=True) as loaded:
                probe_reps_inp = loaded['probe_reps_inp']
                probe_reps_out = loaded['probe_reps_out']
                print(probe_reps_inp.shape)
                print(probe_reps_out.shape)

            step = int(npz_path.stem[-12:])
            steps.add(step)
            print(f'step={step}')

            # do clustering
            Z = linkage(probe_reps_out, method=METHOD, metric=METRIC)
            # compute cophenetic coefficient - used to check that linkage is good and if data is nested
            c, _ = cophenet(Z, pdist(probe_reps_out, metric=METRIC))
            print(f'cophenetic coefficient={c:.2f}')

            # collect cophenetic coefficient
            param_name2c_mat.setdefault(param_path.name, np.zeros((len(job_names), len(npz_paths))))
            param_name2c_mat[param_path.name][row_id][col_id] = c
            print(param_name2c_mat[param_path.name].round(2))

            # plot
            if PLOT_DENDROGRAM:
                fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
                plt.title(f'Hierarchical Clustering Dendrogram (truncated)'
                          f'\nprobe representations at output'
                          f'\nlinkage method={METHOD}'
                          f'\nstep={step:,}'
                          f'\ncophenetic coeff.={c:.2f}')
                ax.set_xlabel('')
                ax.set_ylabel(f'{METRIC} distance')
                dendrogram(
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
                plt.show()

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


# TODO plot cophenetic coefficient across training
fig = make_summary_fig(summaries,
                       ylabel='Cophenetic Coefficient' + '\n+/- 95%-CI',
                       figsize=(6, 4),
                       ylims=[0, 1],
                       )
fig.show()