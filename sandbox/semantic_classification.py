"""
How much information is there in the left-context or right-context of a target word about semantic category membership?

The LDA classifier achieves 100% accuracy with default prams and no scaling.
The Logistic classifier achieves above 90% accuracy only if input features are scaled.
Perfect performance is achieved because there are only 700 targets and there are many more features (5k),
so the classifier can remember the training data perfectly.
also, these accuracies are obtained when tokens are collapsed to types,
and this makes each data point much more informative than individual tokens


It takes about 600 of the most frequent contexts to get above 95% accuracy and about 1000 to get 99% accuracy.
This is only true when contexts must not be shared across the 2 partitions.
When context sharing is enforced, it takes more than 900 contexts just to get above 90%

"""
from dataclasses import dataclass
from typing import List, Optional
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc
from pyitlib import discrete_random_variable as drv

from aochildes.dataset import AOChildesDataSet

from entropicstarttheory import configs
from entropicstarttheory.io import load_probe2cat
from entropicstarttheory.bpe import train_bpe_tokenizer

BPE_VOCAB_SIZE = 8_000  # this number includes 256 reserved latin and non-latin characters
N_COMPONENTS = 27  # should be number of categories - 1
NUM_X = 5  # the more, the smoother the lines in the figure
STRUCTURE_NAME = 'sem-all'

VERBOSE_COEF = False
VERBOSE_AUC = False
VERBOSE_MUTUAL_INFO = False

if BPE_VOCAB_SIZE < 256:
    raise AttributeError('BPE_VOCAB_SIZE must be greater than 256.')

np.set_printoptions(suppress=True)

num_parts = 2
num_contexts_levels = [int(i) for i in np.linspace(10, 500, NUM_X)]  # approx 100-200


def train_clf(x_: np.array,
              y_: np.array,
              shuffled_labels_: bool,
              ):

    clf_ = LinearDiscriminantAnalysis(n_components=N_COMPONENTS)
    if shuffled_labels_:
        clf_.fit(x_, np.random.permutation(y_))
    else:
        clf_.fit(x_, y_)

    return clf_  # for testing on another partition


def make_x_y(df_: pd.DataFrame,
             contexts_shared_: List[str],
             direction_: str,
             part_id_: int,
             ):

    # get only features in a given partition
    df_part = df_[df_['part_id'] == part_id_]

    # transform data so that columns are contexts, rows are probes, and values are frequencies
    dummies = pd.get_dummies(df_part[direction_])
    dummies['probe'] = df_part['probe']
    dummies.set_index('probe', inplace=True)
    df_x = dummies.groupby('probe').sum()

    # get only features that are shared
    df_x = df_x[contexts_shared_]

    assert df_x.columns.values.tolist() == contexts_shared_
    assert len(df_x.columns.values.tolist()) == len(contexts_shared_)

    # convert data to numeric
    x_ = df_x.values
    y_ = np.array([cat2id[probe2cat[p]] for p in df_x.index])

    # scale (so that features are the proportion of times a probe occurs with a given context)
    x_ = (x_ + 1) / (x_.sum(1)[:, np.newaxis] + 1)

    return x_, y_


def eval_clf(clf_: LinearDiscriminantAnalysis,
             x_: np.array,
             y_: np.array,
             ) -> float:

    # eval on the same data
    acc = clf_.score(x_, y_)
    print(f'accuracy ={acc:.4f}')

    # get tpr and fpr for each category
    prob_mat = clf_.predict_proba(x_)
    cat2roc = {}
    for cat in cat2id:
        fpr, tpr, thresholds = roc_curve(np.where(y_ == cat2id[cat], 1, 0),
                                         prob_mat[:, cat2id[cat]])
        cat2roc[cat] = (fpr, tpr)

    # print AUC for each category (labels and colors are sorted by AUC)
    for cat, (fpr, tpr) in sorted(cat2roc.items(),
                                  key=lambda i: auc(*i[1]),
                                  reverse=True):
        if VERBOSE_AUC:
            print(f'{cat:<12} auc={auc(fpr, tpr)}')

    return acc


# ############################################################## conditions

context_directions = ['l', 'r']
shuffled_labels_levels = [True, False]
transfer_directions = ['f', 'b']  # forward, and backward transfer

# ############################################################## start tokenization

# get corpus
transcripts = AOChildesDataSet().load_transcripts()
text_original = ' '.join(transcripts)
tokens_original = text_original.split()

# load only probes that are in the data
probes_in_data = set()
num_probes = 0
types_in_sentences = set(tokens_original)
probe2cat = load_probe2cat(configs.Dirs.root, STRUCTURE_NAME, 'aochildes')
num_probes += len(probe2cat)
for probe in probe2cat.keys():
    if probe in types_in_sentences:
        probes_in_data.add(probe)
    else:
        print(f'probe={probe:<24} not in original data. Excluded.')
probes = list(probes_in_data)

# make cat2id
categories = set(sorted(probe2cat.values()))
cat2id = {cat: n for n, cat in enumerate(categories)}

# replace probes in corpus with special_token
special_token = '<probe>'
tokens_special = []
replaced_probes = []
for token in tokens_original:
    if token in probes_in_data:
        tokens_special.append(special_token)
        replaced_probes.append(token)
    else:
        tokens_special.append(token)

# tokenize corpus
print(f'Tokenizing {len(transcripts)} transcripts..', flush=True)
special_token = '<probe>'
tokenizer = train_bpe_tokenizer(transcripts, BPE_VOCAB_SIZE, special_tokens=[special_token])
text_special = ' '.join(tokens_special)
tokens: List[str] = [t for t in tokenizer.encode(text_special, add_special_tokens=True).tokens
                     if t not in {'Ġ', '', ' '}]
print(f'{len(set(tokens)):,} types in tokenized text', flush=True)
print(f'Tokenized text has {len(tokens) - len(tokens_original):,} more tokens than before tokenization')

# TODO tokenization produces tokens that resemble probes, but should not be considered probes:
# TODO ('Ġadv', 'ice',), ('Ġpleas', 'ant')
# TODO how to differentiate this "incidental" probes from real probes?

# print(sorted(tokenizer.get_vocab().items(), key=lambda i: i[1]))

# replace special token with original probes
replaced_probes_it = iter(replaced_probes)
tokens = [token if token != special_token else next(replaced_probes_it) for token in tokens]
assert len(list(replaced_probes_it)) == 0

# ############################################################## get partitions and contexts

# make partitions
part_id2tokens = {}
for part_id in range(num_parts):
    partition_size = len(tokens) // num_parts
    start = part_id * partition_size
    part_id2tokens[part_id] = tokens[start:start + partition_size]

# get all contexts
part_id2contexts_l = defaultdict(list)
part_id2contexts_r = defaultdict(list)
for part_id in range(num_parts):
    part = part_id2tokens[part_id]
    for n, token in enumerate(part):
        if token in probe2cat:
            part_id2contexts_l[part_id].append(part[n - 1])
            part_id2contexts_r[part_id].append(part[n + 1])

# get shared contexts
cd2contexts_shared = {'l': set(part_id2contexts_l[0]),
                      'r': set(part_id2contexts_r[0])}
for part_id in range(num_parts):
    cd2contexts_shared['l'].intersection_update(part_id2contexts_l[part_id])
    cd2contexts_shared['r'].intersection_update(part_id2contexts_r[part_id])

# count (only shared) contexts so that they can be excluded by frequency
cd2context2f = {cd: Counter() for cd in context_directions}
for part_id in range(num_parts):
    cd2context2f['l'].update([c for c in part_id2contexts_l[part_id] if c in cd2contexts_shared['l']])
    cd2context2f['r'].update([c for c in part_id2contexts_r[part_id] if c in cd2contexts_shared['r']])

for context_dir, c_shared in cd2contexts_shared.items():
    print(f'direction={context_dir} num contexts={len(c_shared)}')

# ############################################################# collect all data

# collect all data
name2col = {'probe': [],
            'l': [],
            'r': [],
            'cat': [],
            'part_id': []
            }
for part_id, part in part_id2tokens.items():
    for n, token in enumerate(part):
        if token in probe2cat:
            name2col['probe'].append(token)
            name2col['l'].append(part[n - 1])
            name2col['r'].append(part[n + 1])
            name2col['cat'].append(probe2cat[token])
            name2col['part_id'].append(part_id)
df = pd.DataFrame(data=name2col)

# ############################################################## use mutual info to estimate redundancy


# approximate the redundancy between left and right contexts and semantic category membership.
# note: an upper bound on redundancy is the minimum mutual information, min(mi(l,c), mi(r,c))
# Nils Bertschinger, Johannes Rauh, Eckehard Olbrich, and Jürgen Jost (2013).
# Shared information—new insights and problems in decomposing information in complex systems.
for part_id in range(num_parts):

    df_part = df[df['part_id'] == part_id]

    if VERBOSE_MUTUAL_INFO:
        token2id = {token: n for n, token in enumerate(tokens)}
        l_obs = [token2id[token] for token in df_part['l']]
        r_obs = [token2id[token] for token in df_part['r']]
        c_obs = [cat2id[cat] for cat in df_part['cat']]
        nmi_lc = drv.information_mutual_normalised(l_obs, c_obs)
        nmi_rc = drv.information_mutual_normalised(r_obs, c_obs)
        print(f'part_id={part_id} nmi(l,c)               ={nmi_lc:.4f}')
        print(f'part_id={part_id} nmi(r,c)               ={nmi_rc:.4f}')

# ############################################################## train classifier on all conditions


@dataclass
class Fit:
    clf: LinearDiscriminantAnalysis
    context_dir: str
    num_contexts: int
    part_id: int
    shuffled_labels: bool

    # collecting the already processed data speeds evaluation
    x: np.array
    y: np.array


fits = []

# collect data and train
for context_dir in context_directions:

    for part_id in range(num_parts):

        for num_contexts in num_contexts_levels:

            context2f: Counter = cd2context2f[context_dir]
            contexts_shared = [c for c, f in context2f.most_common(num_contexts)]

            for shuffled_labels in shuffled_labels_levels:

                # train
                x, y = make_x_y(df, contexts_shared, context_dir, part_id)
                clf = train_clf(x, y, shuffled_labels)

                # collect classifier, data, and information about condition
                fits.append(Fit(clf=clf,
                                context_dir=context_dir,
                                num_contexts=num_contexts,
                                part_id=part_id,
                                shuffled_labels=shuffled_labels,
                                x=x,
                                y=y
                                ))

# ############################################################## evaluate classifiers


def get_train_condition(cd: str,
                        pi: int,
                        sl: bool
                        ) -> str:
    """ there are 4 conditions, one for each curve that is plotted. the design is 2x2x1"""
    return '-'.join([cd, f'partition{pi+1}', 'shuffled' if sl else ''])


def get_test_condition(cd: str,
                       td: str,
                       sl: bool
                       ) -> str:
    """ there are 4 conditions, one for each curve that is plotted. the design is 2x2x1"""
    return '-'.join([cd, td, 'shuffled' if sl else ''])


train_condition2curve = {get_train_condition(cd, pi, sl): [] for cd, pi, sl in product(context_directions,
                                                                                       range(num_parts),
                                                                                       shuffled_labels_levels)}

test_condition2curve = {get_test_condition(cd, td, sl): [] for cd, td, sl in product(context_directions,
                                                                                     transfer_directions,
                                                                                     shuffled_labels_levels)}

# evaluate classifiers on in-sample data
for fit in fits:

    acc = eval_clf(fit.clf,
                   fit.x,
                   fit.y
                   )

    # collect accuracy in one of 8 curves
    condition = get_train_condition(fit.context_dir, fit.part_id, fit.shuffled_labels)
    train_condition2curve[condition].append(acc)


# evaluate classifiers on out-of-sample data
for fit in fits:

    part_id = {0: 1, 1: 0}[fit.part_id]  # use opposite partition during evaluation

    context2f: Counter = cd2context2f[fit.context_dir]
    contexts_shared = [c for c, f in context2f.most_common(fit.num_contexts)]

    acc = eval_clf(fit.clf,
                   *make_x_y(df, contexts_shared, fit.context_dir, part_id)
                   )

    # collect accuracy in one of 8 curves
    transfer_dir = 'f' if part_id == 1 else 'b'
    condition = get_test_condition(fit.context_dir, transfer_dir, fit.shuffled_labels)
    test_condition2curve[condition].append(acc)

# note: top-left has context_dir=left and in-sample accuracy
# note: top-right has context_dir=right and in-sample accuracy
# note: bottom-left has context_dir=left and out-of-sample accuracy
# note: bottom-right has context_dir=right and out-of-sample accuracy
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 4), dpi=configs.Figs.dpi)

# plot in-sample accuracy in top row of figure
for ax, context_dir in zip(axes[0], context_directions):
    if context_dir == context_directions[0]:
        ax.set_ylabel('In-Sample\nClassification Accuracy',
                      fontsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.yaxis.grid(True)
    ax.set_title(f'context direction = {context_dir}')
    ax.set_ylim([0, 1.0])
    for part_id in range(num_parts):
        for shuffled_labels in shuffled_labels_levels:
            condition = get_train_condition(context_dir, part_id, shuffled_labels)
            curve = train_condition2curve[condition]
            ax.plot(num_contexts_levels,
                    curve,
                    color=f'C{part_id}',
                    linestyle='--' if shuffled_labels else '-',
                    label=condition,
                    )
# plot out-of-sample accuracy in bottom row of figure
for ax, context_dir in zip(axes[1], context_directions):
    ax.set_xlabel('Number of Context Words',
                  fontsize=10)
    if context_dir == context_directions[0]:
        ax.set_ylabel('Out-of-Sample\nClassification Accuracy',
                      fontsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.yaxis.grid(True)
    ax.set_ylim([0, 0.4])
    for transfer_dir in transfer_directions:
        for shuffled_labels in shuffled_labels_levels:
            condition = get_test_condition(context_dir, transfer_dir, shuffled_labels)
            curve = test_condition2curve[condition]
            ax.plot(num_contexts_levels,
                    curve,
                    color=f'C{transfer_directions.index(transfer_dir)}',
                    linestyle='--' if shuffled_labels else '-',
                    label=condition,
                    )

# plt.legend(fontsize=10,
#            frameon=False,
#            # loc='lower center',
#            # ncol=2,
#            )


plt.show()
