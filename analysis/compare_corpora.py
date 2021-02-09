"""
Research question:
how many more/less probe words are in newsela vs. childes?
"""

from preppy import FlexiblePrep
from preppy.docs import load_docs
from categoryeval.probestore import ProbeStore

from childesrnnlm import configs


CORPUS_NAMES = ['childes-20191112', 'childes-20191112-ce-g4']
PROBES_NAME = 'sem-4096'


probe2c2f = {}
cat2c2f = {}

for corpus_name in CORPUS_NAMES:
    num_types = 4096 * 4 if corpus_name == 'newsela' else 4096  # x4 is suitable for newsela

    corpus_path = configs.Dirs.corpora / f'{corpus_name}.txt'
    train_docs, _ = load_docs(corpus_path, num_test_docs=0)

    prep = FlexiblePrep(train_docs,
                        reverse=False,
                        sliding=False,
                        num_types=num_types,
                        num_parts=256,
                        num_iterations=(20, 20),
                        batch_size=64,
                        context_size=7,
                        num_evaluations=20,
                        )
    corpus_name_no_suffix = corpus_name
    for suffix in ['-ce', '-g4']:
        corpus_name_no_suffix = corpus_name_no_suffix.replace(suffix, '')
    probestore = ProbeStore(corpus_name_no_suffix,
                            PROBES_NAME,
                            prep.store.w2id,
                            set(),
                            warn=False)

    # count total probe occurrences in corpus
    num_total = 0
    for p in probestore.types:
        cat = probestore.probe2cat[p]

        num_total += prep.store.w2f[p]

        probe2c2f.setdefault(p, {cn: 0 for cn in CORPUS_NAMES})[corpus_name] += prep.store.w2f[p]
        cat2c2f.setdefault(cat, {cn: 0 for cn in CORPUS_NAMES})[corpus_name] += prep.store.w2f[p]

    print(f'Num probe occurrences in {corpus_name:>32}={num_total:,}')

print()
for p, c2f in probe2c2f.items():
    print(f'{p:<16} {CORPUS_NAMES[0]}: {c2f[CORPUS_NAMES[0]]:>6} {CORPUS_NAMES[1]}: {c2f[CORPUS_NAMES[1]]:>6}')

print()
for cat, c2f in cat2c2f.items():
    print(f'{cat:<16} {CORPUS_NAMES[0]}: {c2f[CORPUS_NAMES[0]]:>6,} {CORPUS_NAMES[1]}: {c2f[CORPUS_NAMES[1]]:>6,}')
