"""
Research question:
is next-word distribution for nouns - aka p(X|noun) - more similar to unconditional p(X)
in partition 1 or 2 of AO-CHILDES?
"""

from pyitlib import discrete_random_variable as drv

from preppy.legacy import TrainPrep
from categoryeval.dp import DPScorer

from startingabstract.docs import load_docs
from startingabstract import config


CORPUS_NAME = 'childes-20191112'
NOUNS_NAME = 'singular-nouns-4096'
VERBS_NAME = 'all-verbs-4096'

corpus_path = config.Dirs.corpora / f'{CORPUS_NAME}.txt'
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

for tokens in [prep.store.tokens[:prep.midpoint],
               prep.store.tokens[-prep.midpoint:]]:
    dp_scorer = DPScorer(CORPUS_NAME,
                         probes_names=[NOUNS_NAME, VERBS_NAME, 'unconditional'],
                         tokens=tokens,
                         types=prep.store.types,
                         num_parts=1,
                         )

    # conditional entropy (how much more information in X when y is known?)
    x = dp_scorer.name2q[NOUNS_NAME]
    y = dp_scorer.name2q['unconditional']
    assert x.shape == y.shape
    print(NOUNS_NAME)
    print(f'xe={drv.entropy_cross_pmf(x, y)}')
    print(f'js={drv.divergence_jensenshannon_pmf(x, y)}')

    # conditional entropy (how much more information in X when y is known?)
    x = dp_scorer.name2q[VERBS_NAME]
    y = dp_scorer.name2q['unconditional']
    assert x.shape == y.shape
    print(VERBS_NAME)
    print(f'xe={drv.entropy_cross_pmf(x, y)}')
    print(f'js={drv.divergence_jensenshannon_pmf(x, y)}')

