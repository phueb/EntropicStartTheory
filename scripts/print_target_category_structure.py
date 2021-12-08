import pandas as pd

from entropicstarttheory.io import load_probe2cat
from entropicstarttheory import configs

probe2cat = load_probe2cat(configs.Dirs.root, structure_name='sem-2021', corpus_name='aochildes')


NUM_TABLES = 3


cat2probes = {cat: [p for p in probe2cat if probe2cat[p] == cat]
              for cat in set(probe2cat.values())}

# sort
cat2probes = {k: v for k, v in sorted(cat2probes.items(), key=lambda i: len(i[1]))}


num_cats = len(cat2probes)
num_cats_in_table = num_cats // NUM_TABLES

for table_id in range(NUM_TABLES):

    cats_in_table = list(cat2probes)[:num_cats_in_table]

    # padding
    data = {}
    max_rows = max([len(cat2probes[c]) for c in cats_in_table])
    padding = [''] * max_rows
    for cat in cats_in_table:
        data[cat] = cat2probes[cat] + padding[:max_rows - len(cat2probes[cat])]

    df = pd.DataFrame(data)
    print(df.to_latex(index=False))

print(f'Number of categories={num_cats}')

assert num_cats_in_table * NUM_TABLES == num_cats