from typing import List, Generator
from collections import Counter

from aochildes.dataset import ChildesDataSet

from entropicstarttheory.io import load_probe2cat
from entropicstarttheory import configs
from entropicstarttheory.bpe import train_bpe_tokenizer


POSITION = 1
NUM_PARTS = 8
NUM_TYPES = 8000

transcripts = ChildesDataSet().load_transcripts()
text_original = ' '.join(transcripts)
tokens_original = text_original.split()
print(f'Loaded {len(tokens_original):,} words.')

# load only probes that are in the data
probes_in_data = set()
num_total = 0
types_in_sentences = set(tokens_original)
probe2cat = load_probe2cat(configs.Dirs.root, structure_name='sem-2021', corpus_name='aochildes')
num_total += len(probe2cat)
for probe in probe2cat.keys():
    if probe in types_in_sentences:
        probes_in_data.add(probe)
    else:
        print(f'probe={probe:<24} not in original data. Excluded.')
print(f'{len(probes_in_data)} of {num_total} total probes occur in original data')
probes = list(probes_in_data)

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

# tokenize text
print(f'Tokenizing {len(transcripts)} transcripts..', flush=True)
tokenizer = train_bpe_tokenizer(transcripts, NUM_TYPES, special_tokens=[special_token])
text_special = ' '.join(tokens_special)
tokens: List[str] = [t for t in tokenizer.encode(text_special, add_special_tokens=True).tokens
                     if t not in {'Ġ', '', ' '}]

# replace special token with original probes
replaced_probes_it = iter(replaced_probes)
tokens = [token if token != special_token else next(replaced_probes_it) for token in tokens]
t2f = Counter(tokens)
assert len(list(replaced_probes_it)) == 0


def gen_pr_left_contexts_by_partition(toks: List[str]) -> Generator[List[str], None, None]:
    """
    find perfect-redundancy left-contexts of probe words.

    such left-contexts are those which only precede one probe word and do not occur anywhere else in the same partition
    """

    num_ts_in_part = len(toks) // NUM_PARTS

    res = []
    pairs = set()  # includes e.g. (left-context, probe word)
    tokens_in_part = []
    for i, ti in enumerate(toks):

        tokens_in_part.append(ti)

        # collect pairs
        if ti in probe2cat:
            try:
                lc = tokens[i - POSITION]
            except IndexError:
                pass
            else:
                pairs.add((lc, ti))

        # yield results only when a whole partition worth of tokens have been processed, then clear results
        if len(tokens_in_part) % num_ts_in_part == 0:

            # find num perfect-redundancy contexts
            for lc, p in pairs:
                num = 0
                for j, tj in enumerate(tokens_in_part):
                    if tj == lc:
                        if tokens_in_part[j + POSITION] != p:
                            break
                        else:
                            num += 1
                else:
                    res.append((lc, p, num))

            yield res
            res = []
            pairs = set()
            tokens_in_part = []


print(f'Looping over {len(tokens)} tokens')
for part_id, data in enumerate(gen_pr_left_contexts_by_partition(tokens)):

    props = []
    for di in data:
        lc, p, num = di
        prop = num / t2f[p]  # what is the proportion of times a probe is preceded by pr left-context
        props.append(prop)
        print(f'{lc:>12} {p:>12} {num:>6} {t2f[p]:>6} {prop:.12f}')

    mean_prop = sum(props) / len(props)
    print(f'Partition {part_id:>6,} | prop={mean_prop} Found {len(data):>6,} perfect-redundancy left-contexts')
