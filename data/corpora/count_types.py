from pathlib import Path

num_types = 4095

# load corpus
corpus_path = Path(__file__).parent / f'corpus_original_{num_types}.txt'
if not corpus_path.exists():
    raise FileNotFoundError(f'Did not find {corpus_path}')
with corpus_path.open('r') as f:
    text_original = f.read().replace('\n', ' ')
tokens_original = text_original.split()

print(f'Loaded {len(tokens_original):,} tokens.')
print(f'Loaded {len(set(tokens_original)):,} types.')

# for token in tokens_original[:100]:
#     print(token)