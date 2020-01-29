"""
Research question:
is the connection between a nouns-next word P less lexically specific (higher conditional entropy) in p1 vs p2?
if so, this would support the idea that nouns are learned more abstractly/flexibly in p1.

conditional entropy (x, y) = how much moe information I need to figure out what X is when Y is known.

so if y is the probability distribution of next-words, and x is P over nouns,
 the hypothesis is that conditional entropy is higher in partition 1 vs. 2

"""

from pyitlib import discrete_random_variable as drv
from numpy.lib.stride_tricks import as_strided
import numpy as np
import matplotlib.pyplot as plt

from categoryeval.ba import BAScorer
from preppy.legacy import TrainPrep

from startingabstract.docs import load_docs
from startingabstract import config


CORPUS_NAME = 'childes-20191112'
NOUNS_NAME = 'sem-4096'
CONTROL = False  # if True, all words are used, instead of just nouns

corpus_path = config.Dirs.corpora / f'{CORPUS_NAME}.txt'
train_docs, _ = load_docs(corpus_path)

prep = TrainPrep(train_docs,
                 reverse=False,
                 num_types=4096,  # TODO
                 num_parts=2,  # TODO
                 num_iterations=[20, 20],
                 batch_size=64,
                 context_size=7,
                 num_evaluations=10,
                 )

ba_scorer = BAScorer(CORPUS_NAME,
                     probes_names=[NOUNS_NAME],
                     w2id=prep.store.w2id)

probes = ba_scorer.name2store[NOUNS_NAME].types
print(f'num probes={len(probes)}')

abc = np.arange(prep.store.num_types)


y_2_3 = []
y_1_2 = []
for token_ids in prep.reordered_parts:

    # windows
    token_ids_array = np.array(token_ids, dtype=np.int64)
    num_possible_windows = len(token_ids_array) - prep.num_tokens_in_window
    shape = (num_possible_windows, prep.num_tokens_in_window)
    windows = as_strided(token_ids_array, shape, strides=(8, 8), writeable=False)
    print(f'Matrix containing all windows has shape={windows.shape}')

    # noun windows
    row_ids = np.isin(windows[:, -2], [prep.store.w2id[w] for w in probes])
    if CONTROL:
        row_ids = np.isin(windows[:, -2], np.arange(prep.store.num_types))
    noun_windows = windows[row_ids]
    print(f'num noun-windows={len(noun_windows)}')

    w1 = noun_windows[:, -3]  # last-context-word
    w2 = noun_windows[:, -2]  # noun
    w3 = noun_windows[:, -1]  # next-word

    # entropy - mi = conditional-entropy
    mi = drv.information_mutual(w2, w3)
    _e = drv.entropy(w2)

    ce1_2 = drv.entropy_conditional(w1, w2, Alphabet_X=abc, Alphabet_Y=abc)
    ce2_3 = drv.entropy_conditional(w2, w3, Alphabet_X=abc, Alphabet_Y=abc)

    print(f'ce1_2={ce1_2}')
    print(f'ce2_3={ce2_3}')
    print()
    y_1_2.append(ce1_2)
    y_2_3.append(ce2_3)

if CONTROL:
    print('Warning: CONTROL is True')


fontsize = 12
fig, ax = plt.subplots(1, figsize=(6, 6), dpi=None)
plt.title('', fontsize=fontsize)
ax.set_ylabel('Conditional Entropy(noun|next-word)', fontsize=fontsize)
ax.set_xlabel('AO-CHILDES Partition', fontsize=fontsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
ax.plot(y_1_2, linewidth=2, label='context<-noun')
ax.plot(y_2_3, linewidth=2, label='noun<-next-word')
plt.legend()
plt.tight_layout()
plt.show()