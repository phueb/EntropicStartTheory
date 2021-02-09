"""
Research question:
is next-word distribution for nouns - aka p(X|noun) - more similar to unconditional p(X)
in partition 1 or 2 of AO-CHILDES?
"""

from pyitlib import discrete_random_variable as drv

from preppy import FlexiblePrep
from preppy.docs import load_docs
from categoryeval.dp import DPScorer

from childesrnnlm import configs


CORPUS_NAME = 'newsela'  # 'childes-20191112'
NOUNS_NAME = 'sem-4096'
NUM_TYPES = 4096 * 4 if CORPUS_NAME == 'newsela' else 4096  # x8 is suitable for newsela
EXCLUDED_PROBES = ['tapioca', 'weener', 'acorn']

corpus_path = configs.Dirs.corpora / f'{CORPUS_NAME}.txt'
train_docs, _ = load_docs(corpus_path)

prep = FlexiblePrep(train_docs,
                    reverse=False,
                    sliding=False,
                    num_types=NUM_TYPES,
                    num_parts=2,
                    num_iterations=(20, 20),
                    batch_size=64,
                    context_size=7,
                    num_evaluations=20,
                    )

for tokens in [prep.store.tokens[:prep.midpoint],
               prep.store.tokens[-prep.midpoint:]]:
    dp_scorer = DPScorer(CORPUS_NAME,
                         probes_names=(NOUNS_NAME, 'unconditional'),
                         w2id=prep.store.w2id,
                         tokens=tokens,
                         excluded_probes=EXCLUDED_PROBES,  # probes that do not occur in all partitions
                         warn=False,
                         )

    p = dp_scorer.name2p['unconditional']
    q = dp_scorer.name2p[NOUNS_NAME]
    assert p.shape == q.shape
    print(NOUNS_NAME)
    print(f'xe={drv.entropy_cross_pmf(p, q)}')
    print(f'js={drv.divergence_jensenshannon_pmf(p, q)}')
    print(f'kl={drv.divergence_kullbackleibler_pmf(p, q)}')
    print()