import numpy as np
from collections import Counter

from startingcompact.docs import split

NUM_PARTS = 2


# make dictionary mapping part ID to subset of probes - useful for checking which probes are most frequent in which part
split_size = len(tokens) // NUM_PARTS
part2w2f = {n: Counter(tokens_part) for n, tokens_part in enumerate(split(tokens, split_size))}
part2probes = {part: [] for part in range(NUM_PARTS)}
for probe in probes:
    part = np.argmax([part2w2f[part][probe] for part in range(NUM_PARTS)])
    part2probes[part].append(probe)