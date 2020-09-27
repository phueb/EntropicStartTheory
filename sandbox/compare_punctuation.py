"""
Research question:
is teh next-word distribution for nouns or verbs more similar (informationally speaking) to the unconditional
next-word probability distribution?
if so, this may explain why nouns may be learned earlier than verbs.
"""

import matplotlib.pyplot as plt
import numpy as np
from pyitlib import discrete_random_variable as drv

from preppy.latest import Prep
from categoryeval.dp import DPScorer

from provident.docs import load_docs
from provident import configs


CORPUS_NAME = 'childes-20191112'
NOUNS_NAME = 'singular-nouns-4096'
VERBS_NAME = 'all-verbs-4096'

corpus_path = configs.Dirs.corpora / f'{CORPUS_NAME}.txt'
_train_docs, _ = load_docs(corpus_path)

for keep_punctuation in [True, False]:
    if not keep_punctuation:
        print('Removing punctuation')
        train_docs = []
        for doc in _train_docs:
            new_doc = doc.replace(' .', '').replace(' !', '').replace(' ?', '')
            train_docs.append(new_doc)
    else:
        train_docs = _train_docs

    train_prep = Prep(train_docs,
                      reverse=False,
                      num_types=4096,
                      slide_size=3,
                      batch_size=64,
                      context_size=7,
                      num_evaluations=10,
                      )

    dp_scorer = DPScorer(CORPUS_NAME,
                         probes_names=(NOUNS_NAME, VERBS_NAME, 'unconditional'),
                         tokens=train_prep.store.tokens,
                         types=train_prep.store.types,
                         num_parts=1,
                         )

    # fig
    fig, ax = plt.subplots(figsize=(6, 4), dpi=configs.Figs.dpi)
    plt.title(f'Next-word probability distribution\npunctuation={keep_punctuation}')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(ymax=0.3)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.set_xlabel('Next-word ID (frequency-sorted)')
    ax.set_ylabel('Next-word Probability')
    sorted_w_ids = [train_prep.store.w2id[w]
                    for w in sorted(train_prep.store.types,
                                    key=train_prep.store.w2f.get, reverse=True)]
    # plot q (the next-word distribution conditioned on some category)
    name2sorted_q = {}
    for probes_name in dp_scorer.probes_names:
        _q = dp_scorer.name2p[probes_name]
        sorted_q = [_q[w_id] for w_id in sorted_w_ids]
        ax.semilogx(sorted_q, label=probes_name)

        # collect to compute information theoretic measures
        name2sorted_q[probes_name] = sorted_q

    plt.legend(frameon=False)
    plt.show()

    # how different is noun next-word distribution from unconditional distribution?
    p = dp_scorer.name2p['unconditional']
    q = dp_scorer.name2p[NOUNS_NAME]
    assert p.shape == q.shape
    print(NOUNS_NAME)
    print(f'xe={drv.entropy_cross_pmf(p, q)}')
    print(f'js={drv.divergence_jensenshannon_pmf(p, q)}')
    print(f'kl={drv.divergence_kullbackleibler_pmf(p, q)}')

    # how different is verb next-word distribution from unconditional distribution?
    if False:
        p = dp_scorer.name2p['unconditional']
        q = dp_scorer.name2p[VERBS_NAME]
        assert p.shape == q.shape
        print(VERBS_NAME)
        print(f'xe={drv.entropy_cross_pmf(p, q)}')
        print(f'js={drv.divergence_jensenshannon_pmf(p, q)}')
