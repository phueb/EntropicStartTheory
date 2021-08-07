import pyprind
import torch
import numpy as np
from typing import List, Union, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from torch.nn import CrossEntropyLoss
from pyitlib import discrete_random_variable as drv

from categoryeval.probestore import ProbeStore
from categoryeval.ba import BAScorer
from categoryeval.dp import DPScorer
from categoryeval.cs import CSScorer
from categoryeval.si import SIScorer
from categoryeval.sd import SDScorer

from preppy import Prep

from childesrnnlm import configs
from childesrnnlm.rnn import RNN
from childesrnnlm.representation import make_representations_without_context
from childesrnnlm.representation import make_representations_with_context
from childesrnnlm.representation import make_output_representations
from childesrnnlm.representation import softmax


def calc_perplexity(model: RNN,
                    criterion: CrossEntropyLoss,
                    prep: Prep,
                    is_test: bool,
                    ):
    print(f'Calculating perplexity...')

    pp_sum = 0
    num_batches = 0
    pbar = pyprind.ProgBar(prep.num_mbs, stream=1)

    for windows in prep.generate_batches(is_test=is_test):

        # to tensor
        x, y = np.split(windows, [prep.context_size], axis=1)
        inputs = torch.cuda.LongTensor(x)
        targets = torch.cuda.LongTensor(np.squeeze(y))

        # calc pp (using torch only, on GPU)
        logits = model(inputs)['logits']  # initial hidden state defaults to zero if not provided
        loss_batch = criterion(logits, targets).detach()  # detach to prevent saving complete graph for every sample
        pp_batch = torch.exp(loss_batch)  # need base e

        pbar.update()

        pp_sum += pp_batch.detach().cpu().numpy()
        num_batches += 1
    pp = pp_sum / num_batches
    return pp


def update_pp_performance(performance,
                          model: RNN,
                          criterion: CrossEntropyLoss,
                          prep: Prep,
                          ):
    if configs.Eval.train_pp:
        train_pp = calc_perplexity(model, criterion, prep, is_test=False)
        performance['train_pp'].append(train_pp)
    if configs.Eval.min_num_test_tokens > 0:
        test_pp = calc_perplexity(model, criterion, prep, is_test=True)
        performance['test_pp'].append(test_pp)

    return performance


def update_ma_performance(performance,
                          model: RNN,
                          prep: Prep,
                          structure2probe2cat: Dict[str, Dict[str, str]],
                          ):
    """
    compute average magnitude of probe representations.
    """
    for structure_name in configs.Eval.structures:
        probe2cat = structure2probe2cat[structure_name]
        probe_store = ProbeStore(probe2cat)

        # get probe representations
        probe_token_ids = [prep.token2id[token] for token in probe_store.types]
        probe_reps_inp = make_representations_without_context(model, probe_token_ids)

        ma = np.linalg.norm(probe_reps_inp, axis=1).mean()  # computes magnitude for each vector, then mean
        performance.setdefault(f'ma_n_{structure_name}', []).append(ma)

    return performance


def update_ra_performance(performance,
                          model: RNN,
                          prep: Prep,
                          structure2probe2cat: Dict[str, Dict[str, str]],
                          offset_magnitude: float = 0.001,
                          ):
    """
    compute raggedness of input-output mapping.

    Raggedness is defined as teh divergence between two probability distributions (output by some model)
    given two inputs that are nearby in the input space.

    intuition: if two inputs (which are nearby) produce large divergences at the output,
    it can be said that the model's input-output mapping is ragged.
    """
    for structure_name in configs.Eval.structures:
        probe2cat = structure2probe2cat[structure_name]
        probe_store = ProbeStore(probe2cat)

        # get probe representations
        probe_token_ids = [prep.token2id[token] for token in probe_store.types]
        probe_reps_inp = make_representations_without_context(model, probe_token_ids)
        probe_reps_out = make_output_representations(model, probe_store.types, prep)

        ra_total = 0
        for probe_rep_inp, probe_rep_out in zip(probe_reps_inp, probe_reps_out):

            probe_rep_inp_tiled = np.tile(probe_rep_inp, (len(probe_rep_inp), 1))
            probe_rep_out_tiled = np.tile(probe_rep_out, (len(probe_rep_inp), 1))

            # get vectors representing locations close to probe representations.
            # note: these locations are created by adding offset to a single dimension

            # TODO fine-tune offset magnitude

            offset = np.eye(probe_rep_inp_tiled.shape[0], probe_rep_inp_tiled.shape[1]) * offset_magnitude
            nearby_reps_inp = probe_rep_inp_tiled + offset

            # get outputs for nearby_reps
            with torch.no_grad():
                reshaped = nearby_reps_inp[:, np.newaxis, :].astype(np.float32)  # embeddings must be in last (3rd dim)
                embedded = torch.from_numpy(reshaped).cuda()
                encoded, _ = model.encode(embedded)
                last_encodings = torch.squeeze(encoded[:, -1])
                logits = model.project(last_encodings).cpu().numpy().astype(np.float64)
                nearby_reps_out = softmax(logits)

            ps = probe_rep_out_tiled
            qs = nearby_reps_out

            # calc score for a single probe and collect
            assert ps.shape == qs.shape

            rand_ids = np.random.choice(len(ps), configs.Eval.ra_max_rows, replace=False)
            ps = ps[rand_ids]
            qs = qs[rand_ids]

            ra_probe = drv.divergence_jensenshannon_pmf(ps, qs, base=2, cartesian_product=False).mean()
            ra_total += ra_probe

        ra = ra_total / len(probe_store.types)
        performance.setdefault(f'ra_n_{structure_name}', []).append(ra)

    return performance


def update_ba_performance(performance,
                          model: RNN,
                          prep: Prep,
                          structure2probe2cat: Dict[str, Dict[str, str]],
                          ):
    for structure_name in configs.Eval.structures:
        probe2cat = structure2probe2cat[structure_name]
        ba_scorer = BAScorer(probe2cat)

        probe_store = ba_scorer.probe_store
        probe_token_ids = [prep.token2id[token] for token in probe_store.types]

        probe_reps_o = make_representations_with_context(model, probe_token_ids, prep)
        probe_reps_n = make_representations_without_context(model, probe_token_ids)

        assert len(probe_reps_o) > 0
        assert len(probe_reps_n) > 0

        probe_sims_o = cosine_similarity(probe_reps_o)
        probe_sims_n = cosine_similarity(probe_reps_n)

        ba_n, best_th_n = ba_scorer.calc_score(probe_sims_n, probe_store.gold_sims, 'ba', return_threshold=True)
        ba_o, best_th_o = ba_scorer.calc_score(probe_sims_o, probe_store.gold_sims, 'ba', return_threshold=True)

        performance.setdefault(f'ba_n_{structure_name}', []).append(ba_n)
        performance.setdefault(f'ba_o_{structure_name}', []).append(ba_o)

        # also save best threshold
        performance.setdefault(f'th_n_{structure_name}', []).append(best_th_n)
        performance.setdefault(f'th_o_{structure_name}', []).append(best_th_o)

    return performance


def update_dp_performance(performance,
                          model: RNN,
                          prep: Prep,
                          structure2probe2cat: Dict[str, Dict[str, str]],
                          ):
    """
    calculate distance-to-prototype (aka dp):
    a home-made quantity that is proportional to the distance of all probe representations
    to a location in representational space that can be thought of as the category's "prototype",
    obtained by leveraging all of the co-occurrence data in a corpus.
    """
    for structure_name in configs.Eval.structures:
        probe2cat = structure2probe2cat[structure_name]
        dp_scorer = DPScorer(probe2cat, prep.tokens)

        # get representations at output layer
        probes = dp_scorer.probe_store.types
        qs = make_output_representations(model, probes, prep)

        # check predictions
        max_ids = np.argsort(qs.mean(axis=0))
        print(f'{structure_name} predict:', [prep.types[i] for i in max_ids[-10:]])

        # dp
        dp = dp_scorer.calc_dp(qs, return_mean=True, metric='js')
        performance.setdefault(f'dp_n_{structure_name}', []).append(dp)

    return performance


def update_du_performance(performance,
                          model: RNN,
                          prep: Prep,
                          structure2probe2cat: Dict[str, Dict[str, str]],
                          ):
    """
    calculate distance-to-unigram (aka du):
    a home-made quantity that is proportional to the distance of all probe representations
    to the representation of the unigram-prototype (given by the unigram probability distribution).
    """
    for structure_name in configs.Eval.structures:
        probe2cat = structure2probe2cat[structure_name]
        du_scorer = DPScorer(probe2cat, prep.tokens)

        # collect dp for probes who tend to occur most frequently in some part of corpus
        probes = du_scorer.probe_store.types
        qs = make_output_representations(model, probes, prep)

        # check predictions
        max_ids = np.argsort(qs.mean(axis=0))
        print(f'{structure_name} predict:', [prep.types[i] for i in max_ids[-10:]])

        # calc du
        du = du_scorer.calc_dp(qs, return_mean=True, metric='js', prototype_is_unigram_distribution=True)
        performance.setdefault(f'du_n_{structure_name}', []).append(du)

    return performance


def update_ws_performance(performance,
                          model: RNN,
                          prep: Prep,
                          structure2probe2cat: Dict[str, Dict[str, str]],
                          ):
    """
    compute spread between members of the same category (Within-category Spread).
    """
    for structure_name in configs.Eval.structures:
        probe2cat = structure2probe2cat[structure_name]
        cs_scorer = CSScorer(probe2cat)

        # compute cs for each category
        ws_total = 0
        for cat in cs_scorer.probe_store.cats:
            # get output representations for probes in same category
            ps = make_output_representations(model, cs_scorer.probe_store.cat2probes[cat], prep)
            # compute divergences between exemplars within a category
            ws_cat = cs_scorer.calc_score(ps, ps,
                                          metric=configs.Eval.cs_metric,
                                          max_rows=configs.Eval.ws_max_rows)
            print(f'within-category spread for cat={cat:<18} ={ws_cat:.4f}', flush=True)
            ws_total += ws_cat

        ws = ws_total / len(cs_scorer.probe_store.cats)
        performance.setdefault(f'ws_n_{structure_name}', []).append(ws)

    return performance


def update_as_performance(performance,
                          model: RNN,
                          prep: Prep,
                          structure2probe2cat: Dict[str, Dict[str, str]],
                          ):
    """
    compute spread between members of different categories (Across-category Spread).

    """
    for structure_name in configs.Eval.structures:
        probe2cat = structure2probe2cat[structure_name]
        cs_scorer = CSScorer(probe2cat)

        # get probe representations.
        # note: this function also casts from float32 to float64 to avoid very slow check for NaNs
        probe_reps_out = make_output_representations(model, cs_scorer.probe_store.types, prep)

        # compute all pairwise divergences
        res = cs_scorer.calc_score(probe_reps_out, probe_reps_out,
                                   metric=configs.Eval.cs_metric,
                                   max_rows=configs.Eval.as_max_rows)

        performance.setdefault(f'as_n_{structure_name}', []).append(res)

    return performance


def update_di_performance(performance,
                          model: RNN,
                          prep: Prep,
                          structure2probe2cat: Dict[str, Dict[str, str]],
                          ):
    """
    compute average pairwise Euclidean ("ed") and cosine distance ("cd") between probe representations.
    """
    for structure_name in configs.Eval.structures:
        probe2cat = structure2probe2cat[structure_name]
        ba_scorer = BAScorer(probe2cat)  # we only need this to get the probe_store

        # get probe representations
        probe_store = ba_scorer.probe_store
        probe_token_ids = [prep.token2id[token] for token in probe_store.types]
        probe_reps_n = make_representations_without_context(model, probe_token_ids)

        # compute distances
        ed = euclidean_distances(probe_reps_n, probe_reps_n).mean()
        cd = cosine_distances(probe_reps_n, probe_reps_n).mean()

        performance.setdefault(f'ed_n_{structure_name}', []).append(ed)
        performance.setdefault(f'cd_n_{structure_name}', []).append(cd)

    return performance


def update_si_performance(performance,
                          model: RNN,
                          prep: Prep,
                          structure2probe2cat: Dict[str, Dict[str, str]],
                          ):
    """
    compute silhouette scores.
    how well do probe representations cluster with representations of probes in the same class?
    """
    for structure_name in configs.Eval.structures:
        probe2cat = structure2probe2cat[structure_name]
        si_scorer = SIScorer(probe2cat)

        probe_token_ids = [prep.token2id[token] for token in si_scorer.probe_store.types]
        probe_reps_n = make_representations_without_context(model, probe_token_ids)
        probe_reps_o = make_representations_with_context(model, probe_token_ids, prep)
        cat_ids = [si_scorer.probe_store.cat2id[si_scorer.probe_store.probe2cat[p]]
                   for p in si_scorer.probe_store.types]

        # compute silhouette score
        performance.setdefault(f'si_n_{structure_name}', []).append(
            si_scorer.calc_si(probe_reps_n, cat_ids))
        performance.setdefault(f'si_o_{structure_name}', []).append(
            si_scorer.calc_si(probe_reps_o, cat_ids))

    return performance


def update_sd_performance(performance,
                          model: RNN,
                          prep: Prep,
                          structure2probe2cat: Dict[str, Dict[str, str]],
                          ):
    """
    compute S-Dbw score.
    how well do probe representations cluster with representations of probes in the same class?
    """
    for structure_name in configs.Eval.structures:
        probe2cat = structure2probe2cat[structure_name]
        sd_scorer = SDScorer(probe2cat)

        probe_token_ids = [prep.token2id[token] for token in sd_scorer.probe_store.types]

        probe_reps_n = make_representations_without_context(model, probe_token_ids)
        probe_reps_o = make_representations_with_context(model, probe_token_ids, prep)
        cat_ids = [sd_scorer.probe_store.cat2id[sd_scorer.probe_store.probe2cat[p]]
                   for p in sd_scorer.probe_store.types]

        # compute score
        performance.setdefault(f'sd_n_{structure_name}', []).append(
            sd_scorer.calc_sd(probe_reps_n, cat_ids))
        performance.setdefault(f'sd_o_{structure_name}', []).append(
            sd_scorer.calc_sd(probe_reps_o, cat_ids))

    return performance


def update_pi_performance(performance,
                          model: RNN,
                          prep: Prep,
                          structure2probe2cat: Dict[str, Dict[str, str]],
                          ):
    """
    compute how far the prototype at the input layer is from the origin.

    the prototype at the input layer is computed by first computing the prototype at the output layer,
    and retrieving from a random set of input vectors,
     the one vector that best approximates the prototype at the output.
    """
    for structure_name in configs.Eval.structures:
        probe2cat = structure2probe2cat[structure_name]
        dp_scorer = DPScorer(probe2cat, prep.tokens)

        # compute prototype at output layer
        p = dp_scorer._make_p(is_unconditional=False)

        # find input vector that best approximates prototype at output

        # 1. compute outputs given a sample of random input vectors
        sample_shape = (configs.Eval.pi_num_samples, model.hidden_size)
        random_sample = np.random.uniform(0, configs.Eval.pi_max_magnitude, sample_shape)
        reshaped = random_sample[:, np.newaxis, :].astype(np.float32)  # embeddings must be in last (3rd dim)
        with torch.no_grad():
            embedded = torch.from_numpy(reshaped).cuda()
            encoded, _ = model.encode(embedded)
            last_encodings = torch.squeeze(encoded[:, -1])
            logits = model.project(last_encodings).cpu().numpy().astype(np.float64)
            qs = softmax(logits)

        # 2. compute divergences from prototype for each output
        divergences = [drv.entropy_cross_pmf(p, q) for q in qs]

        # 3. get vector that resulted in smallest divergence at output
        prototype_at_input = random_sample[np.argmin(divergences)]

        # compute score
        origin = np.zeros(model.hidden_size, dtype=np.float64)
        pi = np.asscalar(euclidean_distances(origin.reshape(1, -1), prototype_at_input.reshape(1, -1)))
        performance.setdefault(f'pi_n_{structure_name}', []).append(pi)

    return performance


def get_weights(model):
    ih = model.rnn.weight_ih_l  # [hidden_size, input_size]
    hh = model.rnn.weight_hh_l  # [hidden_size, hidden_size]
    return {'ih': ih, 'hh': hh}