import numpy as np
import torch
from typing import Union

from preppy import FlexiblePrep

from childesrnnlm import configs
from childesrnnlm.rnn import RNN


def make_representations_without_context(model, word_ids):
    """
    make word representations without context by retrieving embeddings
    """
    vocab_reps = model.embed.weight.detach().cpu().numpy()
    probe_reps_n = vocab_reps[word_ids]
    return probe_reps_n


def make_representations_with_context(model: RNN,
                                      token_ids,
                                      prep: FlexiblePrep,
                                      verbose=False,
                                      ) -> np.array:
    """
    make word representations by averaging over all contextualized representations
    """
    all_windows = prep.reordered_windows

    num_words = len(token_ids)
    probe_reps_o = np.zeros((num_words, model.hidden_size))
    for n, token_id in enumerate(token_ids):
        bool_idx = np.isin(all_windows[:, -2], token_id)
        x = all_windows[bool_idx][:, :-1]

        # TODO does this matter?
        if len(x) > configs.Eval.max_num_exemplars:
            x = x[np.random.choice(len(x), size=configs.Eval.max_num_exemplars)]

        inputs = torch.cuda.LongTensor(x)
        num_exemplars, dim1 = inputs.shape
        assert dim1 == prep.context_size, (inputs.shape, x.shape, prep.context_size)
        if verbose:
            print(f'Made {num_exemplars:>6} representations for {prep.types[token_id]:<12}')
        probe_exemplar_reps = model(inputs)['last_encodings'].detach().cpu().numpy()  # [num exemplars, hidden_size]
        probe_reps_o[n] = probe_exemplar_reps.mean(axis=0)
    return probe_reps_o


def make_output_representation(model: RNN,
                               probes,
                               prep: FlexiblePrep,
                               ) -> np.array:
    w_ids = [prep.token2id[w] for w in probes]
    x = np.expand_dims(np.array(w_ids), axis=1)
    inputs = torch.cuda.LongTensor(x)
    logits = model(inputs)['logits'].detach().cpu().numpy()
    res = softmax(logits)
    return res


def softmax(z) -> np.array:
    a = 1  # should be 1 if rows should sum to 1
    z_norm = np.exp(z - np.max(z, axis=a, keepdims=True))
    res = np.divide(z_norm, np.sum(z_norm, axis=a, keepdims=True))

    # check that softmax works correctly - row sum must be close to 1
    assert round(res[0, :].sum().item(), 2) == 1

    return res