"""
How much information is there in the left-context or right-context of a target word about semantic category membership?

In this script, I compute an out-of-sample (i.e cross-validation) accuracy,
not by splitting the corpus data, but by splitting rows in the df (splitting probe words).
this allows me to train the classifier on all the data (fully resolved context distributions),
and to test the knowledge of the LDA that has seen the full dataset.

"""
from dataclasses import dataclass
from typing import List
from itertools import product
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc

from aochildes.dataset import AOChildesDataSet

from entropicstarttheory import configs
from entropicstarttheory.io import load_probe2cat
from entropicstarttheory.bpe import train_bpe_tokenizer

NUM_HELD_OUT = 10  # the number of held_out test sets for each train set
BPE_VOCAB_SIZE = 8_000  # this number includes 256 reserved latin and non-latin characters
N_COMPONENTS = 27  # should be number of categories - 1
NUM_X = 20  # the more, the smoother the lines in the figure
STRUCTURE_NAME = 'sem-all'
INCLUDE_SHUFFLED_LABELS_CONDITION = True

VERBOSE_COEF = False
VERBOSE_AUC = False
VERBOSE_MUTUAL_INFO = False

if BPE_VOCAB_SIZE < 256:
    raise AttributeError('BPE_VOCAB_SIZE must be greater than 256.')

np.set_printoptions(suppress=True)

matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
matplotlib.rcParams['font.family'] = "sans-serif"

num_contexts_levels = [int(i) for i in np.linspace(10, 500, NUM_X)]


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


def make_x_y(df_x_train_: pd.DataFrame,
             contexts_shared_: List[str],
             ):

    # scale before feature selection so that values are the same no matter what features are selected
    # df_x = (df_x.T / df_x.T.sum()).T

    # get only features that are shared
    df_x_train_ = df_x_train_[contexts_shared_]

    assert df_x_train_.columns.values.tolist() == contexts_shared_
    assert len(df_x_train_.columns.values.tolist()) == len(contexts_shared_)

    # convert data to numeric
    x_ = df_x_train_.values
    y_ = np.array([cat2id[probe2cat[p]] for p in df_x_train_.index])

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

# ############################################################## start tokenization

# get corpus
sentences = AOChildesDataSet().load_sentences()
text_original = ' '.join(sentences)
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
print(f'Tokenizing {len(sentences)} transcripts..', flush=True)
special_token = '<probe>'
tokenizer = train_bpe_tokenizer(sentences, BPE_VOCAB_SIZE, special_tokens=[special_token])
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

# get all contexts
contexts_l = []
contexts_r = []

for n, token in enumerate(tokens):
    if token in probe2cat:
        contexts_l.append(tokens[n - 1])
        contexts_r.append(tokens[n + 1])

# get shared contexts
cd2contexts_shared = {'l': set(contexts_l),
                      'r': set(contexts_r)}

# count (only shared) contexts so that they can be excluded by frequency
cd2context2f = {'l': Counter([c for c in contexts_l if c in cd2contexts_shared['l']]),
                'r': Counter([c for c in contexts_r if c in cd2contexts_shared['r']])
                }

for context_dir, c_shared in cd2contexts_shared.items():
    print(f'direction={context_dir} num contexts={len(c_shared)}')

# ############################################################# collect all data

# collect all data
name2col = {'probe': [],
            'l': [],
            'r': [],
            'cat': [],
            }
for n, token in enumerate(tokens):
    if token in probe2cat:
        name2col['probe'].append(token)
        name2col['l'].append(tokens[n - 1])
        name2col['r'].append(tokens[n + 1])
        name2col['cat'].append(probe2cat[token])
df = pd.DataFrame(data=name2col)

# transform data so that columns are contexts, rows are probes, and values are frequencies
cd2df_x = {}
for cd in context_directions:
    dummies = pd.get_dummies(df[cd])
    dummies['probe'] = df['probe']
    dummies.set_index('probe', inplace=True)
    df_x = dummies.groupby('probe').sum()
    df_x = df_x.sample(frac=1.0, replace=False)
    # collect
    cd2df_x[cd] = df_x

# figure out which rows to hold-out for each held-out-id
for cd in context_directions:
    cat_ids_in_index = [cat2id[probe2cat[p]] for p in cd2df_x[cd].index]
    cat_id2f = defaultdict(lambda: 0)
    held_out_ids = []  # this column indicates which rows to hold-out during training
    for cat_id_in_index in cat_ids_in_index:
        held_out_id = cat_id2f[cat_id_in_index]
        held_out_ids.append(held_out_id)
        cat_id2f[cat_id_in_index] += 1
    cd2df_x[cd]['held_out_id'] = held_out_ids

# ############################################################## train classifier on all data


@dataclass
class Fit:
    held_out_id: int
    clf: LinearDiscriminantAnalysis
    context_dir: str
    num_contexts: int
    shuffled_labels: bool

    # collecting the already processed data speeds evaluation
    x: np.array
    y: np.array


fits = []

# collect data and train
for context_dir in context_directions:

    df_x = cd2df_x[context_dir]

    for held_out_id in range(NUM_HELD_OUT):

        # remove rows that should be held-out during training.
        # exactly 28 rows are dropped each time: 1 from each category
        df_x_train = df_x[df_x['held_out_id'] != held_out_id]
        df_x_train.drop('held_out_id', axis=1, inplace=True)

        for num_contexts in num_contexts_levels:

            context2f: Counter = cd2context2f[context_dir]
            contexts_shared = [c for c, f in context2f.most_common(num_contexts)]

            for shuffled_labels in shuffled_labels_levels:

                if shuffled_labels and not INCLUDE_SHUFFLED_LABELS_CONDITION:
                    continue

                # train
                x, y = make_x_y(df_x_train, contexts_shared)
                clf = train_clf(x, y, shuffled_labels)

                # collect classifier, data, and information about condition
                fits.append(Fit(held_out_id=held_out_id,
                                clf=clf,
                                context_dir=context_dir,
                                num_contexts=num_contexts,
                                shuffled_labels=shuffled_labels,
                                x=x,
                                y=y
                                ))

# ############################################################## evaluate classifiers


def get_train_condition(cd: str,
                        sl: bool
                        ) -> str:
    """ there are 4 conditions, one for each curve that is plotted. the design is 2x2x2"""
    return '-'.join([cd, 'shuffled' if sl else ''])


def get_test_condition(cd: str,
                       sl: bool
                       ) -> str:
    """ there are 4 conditions, one for each curve that is plotted. the design is 2x2x2"""
    return '-'.join([cd, 'shuffled' if sl else ''])


train_condition2acc_mat = {get_train_condition(cd, sl): np.zeros((NUM_HELD_OUT, NUM_X))
                           for cd, sl in product(context_directions, shuffled_labels_levels)}

test_condition2acc_mat = {get_test_condition(cd, sl): np.zeros((NUM_HELD_OUT, NUM_X))
                          for cd, sl in product(context_directions, shuffled_labels_levels)}

chance_acc = 0.07

# evaluate classifiers on in-sample data
for fit in fits:

    acc = eval_clf(fit.clf,
                   fit.x,
                   fit.y
                   )

    # collect accuracy
    row_id = fit.held_out_id
    col_id = num_contexts_levels.index(fit.num_contexts)
    condition = get_train_condition(fit.context_dir, fit.shuffled_labels)
    train_condition2acc_mat[condition][row_id, col_id] = acc

    if not INCLUDE_SHUFFLED_LABELS_CONDITION:
        condition = get_train_condition(fit.context_dir, True)
        train_condition2acc_mat[condition][row_id, col_id] = chance_acc


# evaluate classifiers on out-of-sample data (a subset of held-out rows)
for fit in fits:

    # get held-out data
    df_x = cd2df_x[fit.context_dir]
    df_x_test = df_x[df_x['held_out_id'] == fit.held_out_id]
    df_x_test.drop('held_out_id', axis=1, inplace=True)

    context2f: Counter = cd2context2f[fit.context_dir]
    contexts_shared = [c for c, f in context2f.most_common(fit.num_contexts)]

    acc = eval_clf(fit.clf,
                   *make_x_y(df_x_test, contexts_shared)
                   )

    # collect accuracy in one of 8 curves
    row_id = fit.held_out_id
    col_id = num_contexts_levels.index(fit.num_contexts)
    condition = get_test_condition(fit.context_dir, fit.shuffled_labels)
    test_condition2acc_mat[condition][row_id, col_id] = acc

    if not INCLUDE_SHUFFLED_LABELS_CONDITION:
        condition = get_test_condition(fit.context_dir, True)
        test_condition2acc_mat[condition][row_id, col_id] = chance_acc

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
    ax.set_title('left context' if context_dir == 'l' else 'right context ')
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.grid(True)
    ax.set_ylim([0, 1.0])
    for shuffled_labels in shuffled_labels_levels:
        condition = get_train_condition(context_dir, shuffled_labels)
        curve = train_condition2acc_mat[condition].mean(axis=0)
        ax.plot(num_contexts_levels,
                curve,
                # linestyle='--' if shuffled_labels else '-',
                label=condition,
                )
        std_half = train_condition2acc_mat[condition].std(axis=0) / 2
        ax.fill_between(num_contexts_levels,
                        curve + std_half,
                        curve - std_half,
                        alpha=0.2,
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
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.grid(True)
    ax.set_ylim([0, 0.5])
    for shuffled_labels in shuffled_labels_levels:
        condition = get_test_condition(context_dir, shuffled_labels)
        curve = test_condition2acc_mat[condition].mean(axis=0)
        ax.plot(num_contexts_levels,
                curve,
                # linestyle='--' if shuffled_labels else '-',
                label=condition,
                )
        std_half = test_condition2acc_mat[condition].std(axis=0) / 2
        ax.fill_between(num_contexts_levels,
                        curve + std_half,
                        curve - std_half,
                        alpha=0.2,
                        )

# plt.legend(fontsize=10,
#            frameon=False,
#            # loc='lower center',
#            # ncol=2,
#            )


plt.show()
