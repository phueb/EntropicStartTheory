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
PROBES_NAME = 'sem-4096'
CAT = 'NOUN'

NUM_EVALS = 12

corpus_path = config.Dirs.corpora / f'{CORPUS_NAME}.txt'
train_docs, _ = load_docs(corpus_path)

prep = TrainPrep(train_docs,
                 reverse=False,
                 num_types=4096,  # TODO
                 num_parts=128,  # TODO
                 num_iterations=[20, 20],
                 batch_size=64,
                 context_size=7,
                 num_evaluations=10,
                 )

ba_scorer = BAScorer(CORPUS_NAME,
                     probes_names=[PROBES_NAME],
                     w2id=prep.store.w2id)

if PROBES_NAME == 'sem-4096':
    probes = ba_scorer.name2store[PROBES_NAME].types
    CAT = 'NOUN'
else:

    probes = ba_scorer.name2store[PROBES_NAME].cat2probes[CAT]
print(f'num probes={len(probes)}')

abc = np.arange(prep.store.num_types)

# windows
token_ids_array = np.array(prep.store.token_ids, dtype=np.int64)
num_possible_windows = len(token_ids_array) - prep.num_tokens_in_window
shape = (num_possible_windows, prep.num_tokens_in_window)
windows = as_strided(token_ids_array, shape, strides=(8, 8), writeable=False)
print(f'Matrix containing all windows has shape={windows.shape}')

num_windows_list = [int(i) for i in np.linspace(0, len(windows), NUM_EVALS)][1:]


def collect_data(windows, reverse: bool):

    if reverse:
        windows = np.flip(windows, 0)

    y_2_3 = []
    y_1_2 = []
    for num_windows in num_windows_list:

        ws = windows[:num_windows]
        print(num_windows, ws.shape)

        # probe windows
        row_ids = np.isin(ws[:, -2], [prep.store.w2id[w] for w in probes])
        probe_windows = ws[row_ids]
        print(f'num probe windows={len(probe_windows)}')

        w1 = probe_windows[:, -3]  # last-context-word
        w2 = probe_windows[:, -2]  # CAT member
        w3 = probe_windows[:, -1]  # next-word

        # entropy - mi = conditional-entropy
        # mi = drv.information_mutual(w2, w3)
        # _e = drv.entropy(w2)

        ce1_2 = drv.entropy_conditional(w1, w2, Alphabet_X=abc, Alphabet_Y=abc)
        ce2_3 = drv.entropy_conditional(w2, w3, Alphabet_X=abc, Alphabet_Y=abc)

        print(f'ce1_2={ce1_2}')
        print(f'ce2_3={ce2_3}')
        print()
        y_1_2.append(ce1_2)
        y_2_3.append(ce2_3)

    return y_1_2, y_2_3


# collect data
reverse2ys = {}
for reverse, color in zip([False, True], ['C0', 'C1']):
    y_1_2, y_2_3 = collect_data(windows, reverse)
    reverse2ys[reverse] = (y_1_2, y_2_3)

fontsize = 14
x = np.arange(len(num_windows_list)) * num_windows_list[0]

# plot
fig, ax = plt.subplots(1, figsize=(6, 4), dpi=None)
plt.title('', fontsize=fontsize)
ax.set_ylabel(f'Conditional Entropy(prev. word|{CAT})', fontsize=fontsize)
ax.set_xlabel('AO-CHILDES Num Tokens', fontsize=fontsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
ax.plot(x, reverse2ys[False][0], '-', linewidth=2, color='C0', label='age-ordered')
ax.plot(x, reverse2ys[True][0] , '-', linewidth=2, color='C1', label='reverse age-ordered')
plt.legend(frameon=False, fontsize=fontsize, loc='lower right')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, figsize=(6, 4), dpi=None)
plt.title('', fontsize=fontsize)
ax.set_ylabel(f'Conditional Entropy({CAT}|next word)', fontsize=fontsize)
ax.set_xlabel('AO-CHILDES Cumulative Number of Tokens', fontsize=fontsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
ax.plot(x, reverse2ys[False][1], '-', linewidth=2, color='C0', label='age-ordered')
ax.plot(x, reverse2ys[True][1] , '-', linewidth=2, color='C1', label='reverse age-ordered')
plt.legend(frameon=False, fontsize=fontsize, loc='lower right')
plt.tight_layout()
plt.show()



