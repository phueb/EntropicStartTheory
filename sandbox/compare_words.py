"""
Research question:
is next-word distribution for nouns - aka p(X|noun) - more similar to unconditional p(X)
in partition 1 or 2 of AO-CHILDES?
"""

import numpy as np
from pyitlib import discrete_random_variable as drv
from scipy.spatial.distance import jensenshannon

from preppy.legacy import TrainPrep
from categoryeval.dp import DPScorer

from provident.docs import load_docs
from provident import configs


CORPUS_NAME = 'childes-20191112'
NOUNS_NAME = 'singular-nouns-4096'
VERBS_NAME = 'all-verbs-4096'

METRIC = 'js'

corpus_path = configs.Dirs.corpora / f'{CORPUS_NAME}.txt'
train_docs, _ = load_docs(corpus_path)

prep = TrainPrep(train_docs,
                 reverse=False,
                 num_types=4096,
                 num_parts=2,
                 num_iterations=[20, 20],
                 batch_size=64,
                 context_size=7,
                 num_evaluations=10,
                 )

dp_scorer = DPScorer(CORPUS_NAME,
                     probes_names=(NOUNS_NAME, VERBS_NAME, 'unconditional'),
                     tokens=prep.store.tokens,
                     types=prep.store.types,
                     num_parts=1,
                     )

p = dp_scorer.name2p['unconditional']
q = dp_scorer.name2p[NOUNS_NAME]
assert p.shape == q.shape
print(f'js={drv.divergence_jensenshannon_pmf(p, q)}')
print(f'xe={drv.entropy_cross_pmf(p, q)}')

# get predictions "qs"
tmp = []
ct_mat_csr = dp_scorer.ct_mat.tocsr()
probes = dp_scorer.name2probes[NOUNS_NAME]
for probe in probes:
    w_id = prep.store.w2id[probe]
    fs = np.squeeze(ct_mat_csr[w_id].toarray())
    probabilities = fs / fs.sum()
    tmp.append(probabilities)
qs2 = np.array(tmp)

dps = dp_scorer.calc_dp(qs2, 'unconditional', return_mean=False, metric=METRIC)
for probe, dp, probs in sorted(zip(probes, dps, qs2), key=lambda i: i[1], reverse=True):
    next_words = sorted(prep.store.types, key=lambda w: probs[prep.store.w2id[w]].item())[-10:]
    print(f'{probe:<12} {METRIC}={dp:.3f} f={prep.store.w2f[probe]:>9,}', next_words)


