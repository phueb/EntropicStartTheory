from typing import List, Optional

from tokenizers.implementations.byte_level_bpe import ByteLevelBPETokenizer, AddedToken


def train_bpe_tokenizer(sentences: List[str],
                        num_types: int,
                        special_tokens: Optional[List[str]] = None,
                        lowercase: bool = True,
                        ) -> ByteLevelBPETokenizer:
    # only use tokenizer for tokenization, but not vocab
    print(f'Train B-BPE tokenizer with vocab size={num_types:,}', flush=True)
    tokenizer = ByteLevelBPETokenizer(lowercase=lowercase)
    tokenizer.train_from_iterator(sentences,
                                  vocab_size=num_types,
                                  min_frequency=1,
                                  # must set single_word=True
                                  special_tokens=[AddedToken(t, single_word=True) for t in special_tokens],
                                  )
    return tokenizer
