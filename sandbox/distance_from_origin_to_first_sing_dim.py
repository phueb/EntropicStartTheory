import numpy as np
from typing import Optional
from pathlib import Path
from scipy.stats import sem, t
from sklearn.metrics.pairwise import euclidean_distances
from pyitlib import discrete_random_variable as drv
from sklearn.decomposition import NMF

from ludwig.results import gen_param_paths


from entropicstarttheory import __name__
from entropicstarttheory.params import param2default
from entropicstarttheory.figs import make_summary_fig

LUDWIG_DATA_PATH: Optional[Path] = Path('/media/ludwig_data')
RUNS_PATH: Optional[Path] = None  # configs.Dirs.runs if loading runs locally or None if from ludwig

WHERE = ['input', 'output'][1]
FIRST_NUM_STEPS = 100  # the number of first training steps (otherwise, script takes long)
DIM_ID = 0  # zero to use first singular dimension

# param2requests = {'reverse': [True, False]}

# init collection
param_name2y_mat = {}   # matrix has shape (num jobs, num steps)

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

            # compute decomposition, and then distance to first component
            if WHERE == 'input':
                u, s, vt = np.linalg.svd(probe_reps, compute_uv=True)
                sing_vector = s[DIM_ID] * vt[DIM_ID, :].reshape(1, -1)
                origin = np.zeros_like(sing_vector)
                result = euclidean_distances(sing_vector, origin)
            else:
                # we must use NMF because SVD returns components with negative values (invalid as probabilities)
                model = NMF(n_components=1,
                            init='random',
                            random_state=0,
                            solver='mu',
                            beta_loss='kullback-leibler',
                            )
                model.fit(probe_reps)
                comp = model.components_[0]
                p = comp / np.sum(comp)
                ones = np.ones_like(comp)
                q = ones / np.sum(ones)  # technically, this is not the origin
                result = drv.divergence_jensenshannon_pmf(p, q)

            # collect
            param_name2y_mat.setdefault(param_path.name, np.zeros((len(job_names), len(npz_paths))))
            param_name2y_mat[param_path.name][row_id][col_id] = result
            print(param_name2y_mat[param_path.name].round(2))

    # collect summary for each param_name
    job_id = None
    x = np.array(list(sorted(list(steps))))
    y_mean = param_name2y_mat[param_path.name].mean(axis=0)
    y_sem = sem(param_name2y_mat[param_path.name], axis=0)
    confidence = 0.95
    n = len(job_names)
    h = y_sem * t.ppf((1 + confidence) / 2, n - 1)  # margin of error
    summary = (x, y_mean, h, label, job_id)
    summaries.append(summary)  # summary contains: x, mean_y, margin-of-error, label, job_id

if WHERE == 'output':
    y_lims = [0.0, 1]
    y_label = f'JS-Div. between Uniform Distribution and' \
              f'\non-neg. comp.={DIM_ID} of representations at output' \
              f'\n+/- 95%-CI'
else:
    y_lims = [6, 12]
    y_label = f'Euclidean Distance between Origin and' \
              f'\nsing. vector={DIM_ID} of representations at input' \
              f'\n+/- 95%-CI'

fig = make_summary_fig(summaries,
                       ylabel=y_label,
                       figsize=(6, 4),
                       ylims=y_lims,
                       )
fig.show()