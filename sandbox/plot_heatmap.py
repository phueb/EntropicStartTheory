import numpy as np

from preppy.latest import Prep

from startingabstract.docs import load_docs
from startingabstract.figs import plot_heatmap
from startingabstract.memory import set_memory_limit
from startingabstract import config

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20191112'

corpus_path = config.Dirs.corpora / f'{CORPUS_NAME}.txt'
train_docs, _ = load_docs(corpus_path)
prep = Prep(train_docs,
            reverse=False,
            num_types=4096,
            slide_size=3,
            batch_size=64,
            context_size=7,
            num_evaluations=10,
            )


# load nouns (called "members")
nouns = load_probes(f'childes-20191112-nouns')
nouns.intersection_update(prep.store.types)
print(f'num nouns={len(nouns)}')

# load non-nouns (called "non-members")  # TODO do not use random words as most are nouns
verbs = load_probes(f'childes-20180319-verbs')
verbs.intersection_update(prep.store.types)
verbs = verbs[:len(nouns)]
print(f'num rands={len(verbs)}')


# make
set_memory_limit(prop=0.90)
probes = self.name2probes[probes_name]
row_ids = [self.y_words.index(w) for w in probes]
assert row_ids
sliced_ct_mat = self.ct_mat.tocsc()[row_ids, :]

eg_mat, wxs, yws = make_empirical_gold_mat(nouns, verbs, prep, CONTEXT_SIZE, return_labels=True)

# re-order columns
sort_ids = np.argsort(np.mean(eg_mat, axis=0))[::-1]
eg_mat_ordered = eg_mat
max_num_cols = eg_mat_ordered.shape[1]

# plot heatmap
x_tick_labels = np.array(yws)[sort_ids]  # becomes an array of arrays (where inner array was tuple)
y_tick_labels = np.array(nouns)
plot_heatmap(eg_mat_ordered, x_tick_labels, y_tick_labels, dpi=163 * 4)