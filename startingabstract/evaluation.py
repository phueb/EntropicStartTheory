import pyprind
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


from startingabstract import config


def calc_perplexity(model, criterion, prep):
    print(f'Calculating perplexity...')

    pp_sum = torch.tensor(0.0, requires_grad=False)
    num_batches = 0
    pbar = pyprind.ProgBar(prep.num_mbs, stream=1)

    for windows in prep.gen_windows():

        # to tensor
        x, y = np.split(windows, [prep.context_size], axis=1)
        inputs = torch.cuda.LongTensor(x)
        targets = torch.cuda.LongTensor(np.squeeze(y))

        # calc pp (using torch only, on GPU)
        logits = model(inputs)['logits']  # initial hidden state defaults to zero if not provided
        loss_batch = criterion(logits, targets).detach()  # detach to prevent saving complete graph for every sample
        pp_batch = torch.exp(loss_batch)  # need base e

        pbar.update()

        pp_sum += pp_batch
        num_batches += 1
    pp = pp_sum / num_batches
    return pp.item()


def update_pp_metrics(metrics, model, criterion, train_prep, test_prep):
    if not config.Global.train_pp:
        if config.Eval.num_test_docs > 0:
            test_pp = calc_perplexity(model, criterion, test_prep)
            metrics['test_pp'].append(test_pp)
    else:
        train_pp = calc_perplexity(model, criterion, train_prep)
        metrics['train_pp'].append(train_pp)
        if config.Eval.num_test_docs > 0:
            test_pp = calc_perplexity(model, criterion, test_prep)
            metrics['test_pp'].append(test_pp)
    return metrics


def update_ba_metrics(metrics, model, train_prep, ba_scorer):

    for ba_name in ba_scorer.ba_names:

        probe_store = ba_scorer.ba_name2store[ba_name]  # TODO implement ba_scorer

        probe_reps_o = make_probe_reps_o(model, probe_store, train_prep)
        probe_reps_n = make_probe_reps_n(model, probe_store)

        probe_sims_o = cosine_similarity(probe_reps_o)
        probe_sims_n = cosine_similarity(probe_reps_n)

        metrics[config.Metrics.ba_o].append(ba_scorer.calc_score(probe_sims_o, probe_store.gold_sims, 'ba'))
        metrics[config.Metrics.ba_n].append(ba_scorer.calc_score(probe_sims_n, probe_store.gold_sims, 'ba'))

    return metrics


def update_dp_metrics(metrics, model, train_prep, dp_scorer):
    """
    calculate distance-to-prototype (aka dp):
    how well do predictions conform to an abstract prototype?
    """
    for dp_name in dp_scorer.dp_names:
        # collect dp for probes who tend to occur most frequently in some part of corpus
        for part in range(config.Eval.dp_num_parts):
            # predictions_mat
            probes_in_part = dp_scorer.dp_name2part2probes[dp_name][part]
            assert probes_in_part
            w_ids = [train_prep.store.w2id[w] for w in probes_in_part]
            x = np.expand_dims(np.array(w_ids), axis=1)
            inputs = torch.cuda.LongTensor(x)
            logits = model(inputs)['logits'].detach().cpu().numpy()
            predictions_mat = softmax(logits)
            dps = dp_scorer.calc_dp(predictions_mat, dp_name, return_mean=False)

            # check predictions
            max_ids = np.argsort(predictions_mat.mean(axis=0))
            print(f'{dp_name} predict:', [train_prep.store.types[i] for i in max_ids[-10:]])

            # collect
            for pi, dp in enumerate(dps[:3]):  # TODO test individual dp values
                metrics[f'dp_{dp_name}_part{part}_probe{pi}'].append(dp)

    return metrics


def softmax(z):
    a = 1  # should be 1 if rows should sum to 1
    z_norm = np.exp(z - np.max(z, axis=a, keepdims=True))
    res = np.divide(z_norm, np.sum(z_norm, axis=a, keepdims=True))

    # check that softmax works correctly - row sum must be close to 1
    assert round(res[0, :].sum().item(), 2) == 1

    return res


def make_probe_reps_n(model, probe_store):
    """
    make probe representations without context by retrieving embeddings
    """
    vocab_reps = model.embed.weight.detach().cpu().numpy()
    probe_reps_n = vocab_reps[probe_store.vocab_ids]
    return probe_reps_n


def make_probe_reps_o(model, probe_store, train_prep):
    """
    make probe representations by averaging over all contextualized representations
    """
    all_windows = train_prep.reordered_windows

    probe_reps_o = np.zeros((probe_store.num_probes, model.hidden_size))
    for n, vocab_id in enumerate(probe_store.vocab_ids):
        bool_idx = np.isin(all_windows[:, -2], vocab_id)
        x = all_windows[bool_idx][:, :-1]
        inputs = torch.cuda.LongTensor(x)
        num_exemplars, dim1 = inputs.shape
        assert dim1 == train_prep.context_size
        print(f'Made {num_exemplars:>6} representations for {train_prep.store.types[vocab_id]:<12}')
        probe_exemplar_reps = model(inputs)['last_encodings'].detach().cpu().numpy()  # [num exemplars, hidden_size]
        probe_reps_o[n] = probe_exemplar_reps.mean(axis=0)
    return probe_reps_o


def get_weights(model):
    ih = model.rnn.weight_ih_l  # [hidden_size, input_size]
    hh = model.rnn.weight_hh_l  # [hidden_size, hidden_size]
    return {'ih': ih, 'hh': hh}