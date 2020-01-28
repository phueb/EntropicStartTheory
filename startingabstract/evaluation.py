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
        if config.Eval.num_test_docs > 0:
            train_pp = calc_perplexity(model, criterion, train_prep)  # TODO cuda error
            test_pp = calc_perplexity(model, criterion, test_prep)
            metrics['train_pp'].append(train_pp)
            metrics['test_pp'].append(test_pp)
    return metrics


def update_ba_metrics(metrics, model, train_prep, ba_scorer):

    for probes_name in ba_scorer.probes_names:

        probe_store = ba_scorer.name2store[probes_name]  # TODO implement ba_scorer

        probe_reps_o = make_probe_reps_o(model, probe_store, train_prep)
        probe_reps_n = make_probe_reps_n(model, probe_store)

        probe_sims_o = cosine_similarity(probe_reps_o)
        probe_sims_n = cosine_similarity(probe_reps_n)

        metrics[f'ba_o_{probes_name}'].append(ba_scorer.calc_score(probe_sims_o, probe_store.gold_sims, 'ba'))
        metrics[f'ba_n_{probes_name}'].append(ba_scorer.calc_score(probe_sims_n, probe_store.gold_sims, 'ba'))

    return metrics


def update_dp_metrics(metrics, model, train_prep, dp_scorer):  # TODO is this still useful?
    """
    calculate distance-to-prototype (aka dp):
    all divergences are relative to the prototype that best characterizes members belonging to probes_name
    """
    for probes_name in dp_scorer.probes_names:
        # collect dp for probes who tend to occur most frequently in some part of corpus
        for part in range(config.Eval.dp_num_parts):
            # predictions_mat
            probes_in_part = dp_scorer.name2part2probes[probes_name][part]
            assert probes_in_part
            w_ids = [train_prep.store.w2id[w] for w in probes_in_part]
            x = np.expand_dims(np.array(w_ids), axis=1)
            inputs = torch.cuda.LongTensor(x)
            logits = model(inputs)['logits'].detach().cpu().numpy()
            predictions_mat = softmax(logits)

            # check predictions
            max_ids = np.argsort(predictions_mat.mean(axis=0))
            print(f'{probes_name} predict:', [train_prep.store.types[i] for i in max_ids[-10:]])

            # dp
            dp = dp_scorer.calc_dp(predictions_mat, probes_name, metric='js')
            metrics[f'dp_{probes_name}_part{part}_js'].append(dp)

            dp = dp_scorer.calc_dp(predictions_mat, probes_name, metric='xe')
            metrics[f'dp_{probes_name}_part{part}_xe'].append(dp)

    return metrics


def update_dp_metrics_unconditional(metrics, model, train_prep, dp_scorer):
    """
    calculate distance-to-prototype (aka dp):
    all divergences are relative to unconditional prototype, including:
    1. model-based next-word distribution given word
    2. ideal next-word distribution given word
    3. ideal next-word distribution given category of a word
    """
    for probes_name in dp_scorer.probes_names:
        if probes_name == 'unconditional':
            continue

        for part in range(config.Eval.dp_num_parts):

            # qs1
            probes = dp_scorer.name2part2probes[probes_name][part]
            assert probes
            w_ids = [train_prep.store.w2id[w] for w in probes]
            x = np.expand_dims(np.array(w_ids), axis=1)
            inputs = torch.cuda.LongTensor(x)
            logits = model(inputs)['logits'].detach().cpu().numpy()
            qs1 = softmax(logits)

            # qs2
            tmp = []
            ct_mat_csr = dp_scorer.ct_mat.tocsr()
            for p in probes:
                w_id = train_prep.store.w2id[p]
                fs = np.squeeze(ct_mat_csr[w_id].toarray()) + 10e9
                probabilities = fs / fs.sum()
                tmp.append(probabilities)
            qs2 = np.array(tmp)

            # qs3
            qs3 = dp_scorer.name2p[probes_name][np.newaxis, :]

            # dp
            dp1 = dp_scorer.calc_dp(qs1, 'unconditional', metric='js')
            dp2 = dp_scorer.calc_dp(qs2, 'unconditional', metric='js')
            dp3 = dp_scorer.calc_dp(qs3, 'unconditional', metric='js')

            metrics[f'dp_{probes_name}_part{part}_js_unconditional_1'].append(dp1)
            metrics[f'dp_{probes_name}_part{part}_js_unconditional_2'].append(dp2)
            metrics[f'dp_{probes_name}_part{part}_js_unconditional_3'].append(dp3)

            # dp
            dp1 = dp_scorer.calc_dp(qs1, 'unconditional', metric='xe')
            dp2 = dp_scorer.calc_dp(qs2, 'unconditional', metric='xe')
            dp3 = dp_scorer.calc_dp(qs3, 'unconditional', metric='xe')

            metrics[f'dp_{probes_name}_part{part}_xe_unconditional_1'].append(dp1)
            metrics[f'dp_{probes_name}_part{part}_xe_unconditional_2'].append(dp2)
            metrics[f'dp_{probes_name}_part{part}_xe_unconditional_3'].append(dp3)

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


def make_probe_reps_o(model, probe_store, train_prep, verbose=False):
    """
    make probe representations by averaging over all contextualized representations
    """
    all_windows = train_prep.reordered_windows

    probe_reps_o = np.zeros((probe_store.num_probes, model.hidden_size))
    for n, vocab_id in enumerate(probe_store.vocab_ids):
        bool_idx = np.isin(all_windows[:, -2], vocab_id)
        x = all_windows[bool_idx][:, :-1]

        # TODO does this matter?
        # if len(x) > config.Eval.max_num_exemplars:
        #     x = x[np.random.choice(len(x), size=config.Eval.max_num_exemplars)]

        inputs = torch.cuda.LongTensor(x)
        num_exemplars, dim1 = inputs.shape
        assert dim1 == train_prep.context_size
        if verbose:
            print(f'Made {num_exemplars:>6} representations for {train_prep.store.types[vocab_id]:<12}')
        probe_exemplar_reps = model(inputs)['last_encodings'].detach().cpu().numpy()  # [num exemplars, hidden_size]
        probe_reps_o[n] = probe_exemplar_reps.mean(axis=0)
    return probe_reps_o


def get_weights(model):
    ih = model.rnn.weight_ih_l  # [hidden_size, input_size]
    hh = model.rnn.weight_hh_l  # [hidden_size, hidden_size]
    return {'ih': ih, 'hh': hh}