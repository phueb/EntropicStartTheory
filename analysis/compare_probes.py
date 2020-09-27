"""
Research question:
is next-word distribution for nouns - aka p(X|noun) - more similar to unconditional p(X)
in partition 1 or 2 of AO-CHILDES?
"""

import numpy as np
from pyitlib import discrete_random_variable as drv

from preppy import FlexiblePrep
from preppy.docs import load_docs
from categoryeval.dp import DPScorer

from provident import configs


CORPUS_NAME = 'childes-20191112'
PROBES_NAME = 'nouns-singular'

METRIC = 'js'

corpus_path = configs.Dirs.corpora / f'{CORPUS_NAME}.txt'
train_docs, _ = load_docs(corpus_path)

prep = FlexiblePrep(train_docs,
                    reverse=False,
                    sliding=False,
                    num_types=4096,
                    num_parts=2,
                    num_iterations=(20, 20),
                    batch_size=64,
                    context_size=7,
                    num_evaluations=20,
                    )

dp_scorer = DPScorer(CORPUS_NAME,
                     probes_names=(PROBES_NAME, 'unconditional'),
                     tokens=prep.store.tokens,
                     w2id=prep.store.w2id,
                     )

p = dp_scorer.name2p['unconditional']
q = dp_scorer.name2p[PROBES_NAME]
assert p.shape == q.shape
print(f'js={drv.divergence_jensenshannon_pmf(p, q)}')
print(f'xe={drv.entropy_cross_pmf(p, q)}')

# get next-word probability distribution for each probe
tmp = []
ct_mat_csr = dp_scorer.ct_mat.tocsr()
probes = dp_scorer.name2store[PROBES_NAME].types
for probe in probes:
    w_id = prep.store.w2id[probe]
    fs = np.squeeze(ct_mat_csr[w_id].toarray())
    probabilities = fs / fs.sum()
    tmp.append(probabilities)
next_word_probabilities_all = np.array(tmp)

# calc + print divergence from unconditional prototype for each probe (in terms of next-word probability distribution)
dps = dp_scorer.calc_dp(next_word_probabilities_all, 'unconditional', return_mean=False, metric=METRIC)
for probe, dp, ps in sorted(zip(probes, dps, next_word_probabilities_all), key=lambda i: i[1], reverse=True):
    next_words = sorted(prep.store.types, key=lambda w: ps[prep.store.w2id[w]].item())[-10:]
    print(f'{probe:<12} {METRIC}={dp:.3f} f={prep.store.w2f[probe]:>9,}', next_words)

# note: words printed first have next-word probabilities which diverge the most
# from unconditional next-word probabilities


