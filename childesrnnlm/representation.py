import numpy as np
import torch
from typing import List

from preppy import Prep

from childesrnnlm import configs
from childesrnnlm.rnn import RNN


def get_weights(model):
    ih = model.rnn.weight_ih_l  # [hidden_size, input_size]
    hh = model.rnn.weight_hh_l  # [hidden_size, hidden_size]
    return {'ih': ih, 'hh': hh}


def make_inp_representations_without_context(model: RNN,
                                             tokens: List[str],
                                             prep: Prep,
                                             ):
    """
    make word representations without context by retrieving embeddings
    """
    vocab_reps = model.embed.weight.detach().cpu().numpy()

    w_ids = [prep.token2id[w] for w in tokens]
    res = vocab_reps[w_ids]

    # need to cast from float32 to float64 to avoid very slow check for NaNs in drv.divergence_jensenshannon_pmf
    res = res.astype(np.float64)

    return res


def make_inp_representations_with_context(model: RNN,
                                          tokens: List[str],
                                          prep: Prep,
                                          verbose=False,
                                          ) -> np.array:
    """
    make word representations by averaging over all contextualized representations
    """
    all_windows = prep.reordered_windows

    w_ids = [prep.token2id[w] for w in tokens]
    num_words = len(w_ids)
    res = np.zeros((num_words, model.hidden_size))
    for n, token_id in enumerate(w_ids):
        bool_idx = np.isin(all_windows[:, -2], token_id)
        x = all_windows[bool_idx][:, :-1]

        # TODO does this matter?
        if len(x) > configs.Eval.max_num_exemplars:
            x = x[np.random.choice(len(x), size=configs.Eval.max_num_exemplars)]

        inputs = torch.LongTensor(x).cuda()
        num_exemplars, dim1 = inputs.shape
        assert dim1 == prep.context_size, (inputs.shape, x.shape, prep.context_size)
        if verbose:
            print(f'Made {num_exemplars:>6} representations for {prep.types[token_id]:<12}')
        representations = model(inputs)['last_encodings'].detach().cpu().numpy()  # [num exemplars, hidden_size]
        res[n] = representations.mean(axis=0)

    # need to cast from float32 to float64 to avoid very slow check for NaNs in drv.divergence_jensenshannon_pmf
    res = res.astype(np.float64)

    return res


def make_inp_representations(model: RNN,
                             tokens: List[str],
                             prep: Prep,
                             context_type: str,
                             ) -> np.array:
    if context_type == 'n':
        return make_inp_representations_without_context(model, tokens, prep)
    elif context_type == 'o':
        return make_inp_representations_with_context(model, tokens, prep)
    else:
        raise AttributeError('Invalid arg to context_type')


def make_out_representations(model: RNN,
                             tokens: List[str],
                             prep: Prep,
                             context_type: str,
                             ) -> np.array:

    # feed-forward
    if context_type == 'n':
        w_ids = [prep.token2id[w] for w in tokens]
        x = np.expand_dims(np.array(w_ids), axis=1)
        inputs = torch.LongTensor(x).cuda()
        logits = model(inputs)['logits'].detach().cpu().numpy()
    elif context_type == 'o':
        raise NotImplementedError  # TODO implement
    else:
        raise AttributeError('Invalid arg to context_type')

    # softmax (requires 2 dimensions)
    # if only one representation is requested, first dimension is lost in RNN output, so we add it again.
    if len(x) == 1:
        logits = logits[np.newaxis, :]
    res = softmax(logits)

    # need to cast from float32 to float64 to avoid very slow check for NaNs in drv.divergence_jensenshannon_pmf
    res = res.astype(np.float64)

    return res


def softmax(z: np.array) -> np.array:
    assert np.ndim(z) == 2
    a = 1  # should be 1 if rows should sum to 1
    z_norm = np.exp(z - np.max(z, axis=a, keepdims=True))
    res = np.divide(z_norm, np.sum(z_norm, axis=a, keepdims=True))

    # check that softmax works correctly - row sum must be close to 1
    assert round(res[0, :].sum().item(), 2) == 1

    return res
