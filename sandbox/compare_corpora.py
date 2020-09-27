"""
Research question:
how many more/less probe words are in newsela vs. childes?
"""

from preppy import FlexiblePrep
from preppy.docs import load_docs
from categoryeval.probestore import ProbeStore

from provident import configs


CORPUS_NAMES = ['newsela', 'childes-20191112']
PROBES_NAME = 'sem-4096'


p2c2f = {}

for corpus_name in CORPUS_NAMES:
    num_types = 4096 * 8 if corpus_name == 'newsela' else 4096  # x8 is suitable for newsela

    corpus_path = configs.Dirs.corpora / f'{corpus_name}.txt'
    train_docs, _ = load_docs(corpus_path, num_test_docs=0)

    prep = FlexiblePrep(train_docs,
                        reverse=False,
                        sliding=False,
                        num_types=num_types,
                        num_parts=2,
                        num_iterations=(20, 20),
                        batch_size=64,
                        context_size=7,
                        num_evaluations=20,
                        )

    probestore = ProbeStore(corpus_name, PROBES_NAME, prep.store.w2id, set(), warn=False)

    # count total probe occurrences in corpus
    num_total = 0
    for p in probestore.types:
        num_total += prep.store.w2f[p]

        p2c2f.setdefault(p, {})[corpus_name] = prep.store.w2f[p]

    print(f'Num probe occurrences in {corpus_name:>32}={num_total:,}')

for p, c2f in p2c2f.items():
    print(f'{p:<16} {c2f}')
