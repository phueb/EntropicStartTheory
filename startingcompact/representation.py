import numpy as np
import torch

from startingcompact import config


def make_representations_without_context(model, word_ids):
    """
    make word representations without context by retrieving embeddings
    """
    vocab_reps = model.embed.weight.detach().cpu().numpy()
    probe_reps_n = vocab_reps[word_ids]
    return probe_reps_n


def make_representations_with_context(model, word_ids, train_prep, verbose=False):
    """
    make word representations by averaging over all contextualized representations
    """
    all_windows = train_prep.reordered_windows

    num_words = len(word_ids)
    probe_reps_o = np.zeros((num_words, model.hidden_size))
    for n, vocab_id in enumerate(word_ids):
        bool_idx = np.isin(all_windows[:, -2], vocab_id)
        x = all_windows[bool_idx][:, :-1]

        # TODO does this matter?
        if len(x) > config.Eval.max_num_exemplars:
            x = x[np.random.choice(len(x), size=config.Eval.max_num_exemplars)]

        inputs = torch.cuda.LongTensor(x)
        num_exemplars, dim1 = inputs.shape
        assert dim1 == train_prep.context_size, (inputs.shape, x.shape, train_prep.context_size)
        if verbose:
            print(f'Made {num_exemplars:>6} representations for {train_prep.store.types[vocab_id]:<12}')
        probe_exemplar_reps = model(inputs)['last_encodings'].detach().cpu().numpy()  # [num exemplars, hidden_size]
        probe_reps_o[n] = probe_exemplar_reps.mean(axis=0)
    return probe_reps_o


def make_output_representation(model, probes, train_prep):
    w_ids = [train_prep.store.w2id[w] for w in probes]
    x = np.expand_dims(np.array(w_ids), axis=1)
    inputs = torch.cuda.LongTensor(x)
    logits = model(inputs)['logits'].detach().cpu().numpy()
    res = softmax(logits)
    return res


def softmax(z):
    a = 1  # should be 1 if rows should sum to 1
    z_norm = np.exp(z - np.max(z, axis=a, keepdims=True))
    res = np.divide(z_norm, np.sum(z_norm, axis=a, keepdims=True))

    # check that softmax works correctly - row sum must be close to 1
    assert round(res[0, :].sum().item(), 2) == 1

    return res