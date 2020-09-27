"""
Research question:
is the next-word distribution for nouns or verbs more similar to the unconditional
next-word probability distribution?
if so, this may explain why nouns may be learned earlier than verbs.
"""

import matplotlib.pyplot as plt

from preppy import SlidingPrep
from preppy.docs import load_docs
from categoryeval.dp import DPScorer

from provident import configs


CORPUS_NAME = 'newsela'  # 'childes-20191112'
PROBES_NAME = 'sem-4096'
NUM_TYPES = 4096 * 8 if CORPUS_NAME == 'newsela' else 4096  # x8 is suitable for newsela

corpus_path = configs.Dirs.corpora / f'{CORPUS_NAME}.txt'
_train_docs, _ = load_docs(corpus_path, num_test_docs=0)

for keep_punctuation in [True, False]:
    if not keep_punctuation:
        print('Removing punctuation')
        train_docs = []
        for doc in _train_docs:
            new_doc = doc.replace(' .', '').replace(' !', '').replace(' ?', '')
            train_docs.append(new_doc)
    else:
        train_docs = _train_docs

    prep = SlidingPrep(train_docs,
                       reverse=False,
                       num_types=NUM_TYPES,
                       batch_size=64,
                       context_size=7,
                       slide_size=3,
                       num_evaluations=10,
                       )

    dp_scorer = DPScorer(CORPUS_NAME,
                         probes_names=(PROBES_NAME, 'unconditional'),
                         w2id=prep.store.w2id,
                         tokens=prep.store.tokens,
                         warn=False,
                         )

    # fig
    fig, ax = plt.subplots(figsize=(5, 5), dpi=configs.Figs.dpi)
    plt.title(f'Next-word probability distribution\npunctuation={keep_punctuation}')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(ymax=0.2)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.set_xlabel('Next-word ID (frequency-sorted)')
    ax.set_ylabel('Next-word Probability')
    sorted_w_ids = [prep.store.w2id[w]
                    for w in sorted(prep.store.types,
                                    key=prep.store.w2f.get, reverse=True)]
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