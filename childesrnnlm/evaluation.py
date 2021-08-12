import torch
import numpy as np
from typing import List, Union, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
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
    print(f'Calculating {"test" if is_test else "train"} perplexity...', flush=True)

    pp_sum = 0
    num_batches = 0

    with torch.no_grad():

        for windows in prep.generate_batches(is_test=is_test):

            # to tensor
            x, y = np.split(windows, [prep.context_size], axis=1)
            inputs = torch.LongTensor(x).cuda()
            targets = torch.LongTensor(np.squeeze(y)).cuda()

            # calc pp (using torch only, on GPU)
            logits = model(inputs)['logits']  # initial hidden state defaults to zero if not provided
            loss_batch = criterion(logits, targets)
            pp_batch = torch.exp(loss_batch)  # need base e

            pp_sum += pp_batch.cpu().numpy()
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
        probe_reps_inp = make_representations_without_context(model, probe_store.types, prep)

        ma = np.linalg.norm(probe_reps_inp, axis=1).mean()  # computes magnitude for each vector, then mean
        performance.setdefault(f'ma_n_{structure_name}', []).append(ma)

    return performance


def update_ra_performance(performance,
                          model: RNN,
                          prep: Prep,
                          structure2probe2cat: Dict[str, Dict[str, str]],
                          ):
    """


    TODO replace this, and rename

    # TODO compute actual prototype  by using first row in v matrix of input represetnations, and compute its output distribution. compare this to the theoretical prototype -> reverse trained models should score higher on this

    """
    for structure_name in configs.Eval.structures:
        probe2cat = structure2probe2cat[structure_name]
        probe_store = ProbeStore(probe2cat)

        # get probe representations
        probe_reps_out = make_output_representations(model, probe_store.types, prep)

        # compute output that results at origin
        origin = np.zeros((1, model.hidden_size), dtype=np.float32)
        reshaped = origin[:, np.newaxis, :]  # embeddings must be in last (3rd dim)
        with torch.no_grad():
            embedded = torch.from_numpy(reshaped).cuda()
            encoded, _ = model.encode(embedded)
            last_encodings = torch.squeeze(encoded[:, -1])
            logits = model.project(last_encodings).cpu().numpy().astype(np.float64)
            origin_rep_out = softmax(logits[np.newaxis, :])  # softmax requires num dim = 2

        ps = probe_reps_out
        qs = origin_rep_out

        ra = drv.divergence_jensenshannon_pmf(ps, qs, base=2, cartesian_product=True).mean()
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

        # make probe representations
        probe_reps_o = make_representations_with_context(model, ba_scorer.probe_store.types, prep)
        probe_reps_n = make_representations_without_context(model, ba_scorer.probe_store.types, prep)

        assert len(probe_reps_o) > 0
        assert len(probe_reps_n) > 0

        probe_sims_o = cosine_similarity(probe_reps_o)
        probe_sims_n = cosine_similarity(probe_reps_n)

        ba_n, best_th_n = ba_scorer.calc_score(probe_sims_n, ba_scorer.probe_store.gold_sims, 'ba', return_threshold=True)
        ba_o, best_th_o = ba_scorer.calc_score(probe_sims_o, ba_scorer.probe_store.gold_sims, 'ba', return_threshold=True)

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
    calculate distance-to-prototype:
    a home-made quantity that is proportional to the distance of all probe representations
    to a location in representational space that can be thought of as the probe (typically nouns) "prototype",
    obtained by leveraging all of the co-occurrence data in a corpus.

    also, compute distance of output probability due to bias only to prototype
    """
    for structure_name in configs.Eval.structures:
        probe2cat = structure2probe2cat[structure_name]
        dp_scorer = DPScorer(probe2cat, prep.tokens, prep.types)

        # get representations of probes at output
        probes = dp_scorer.probe_store.types
        probe_reps_out = make_output_representations(model, probes, prep)

        # get representation of bias at output
        if model.project.bias is not None:
            projection_bias = model.project.bias.detach().cpu().numpy().astype(np.float64)
            bias_rep_out = softmax(projection_bias[np.newaxis, :])

        # calc divergence between probes and probe prototype
        dp = dp_scorer.score(probe_reps_out, return_mean=True, metric='js')
        performance.setdefault(f'dp_n_{structure_name}', []).append(dp)

        # calc divergence between bias and probe prototype
        if model.project.bias is not None:
            db = dp_scorer.score(bias_rep_out, return_mean=True, metric='js')
            performance.setdefault(f'db_n_{structure_name}', []).append(db)

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
        du_scorer = DPScorer(probe2cat, prep.tokens, prep.types)

        # collect dp for probes who tend to occur most frequently in some part of corpus
        probes = du_scorer.probe_store.types
        qs = make_output_representations(model, probes, prep)

        # calc du
        du = du_scorer.score(qs, return_mean=True, metric='js', prototype_is_unigram_distribution=True)
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

        # compute spread for each category
        ws_total = 0
        for cat in cs_scorer.probe_store.cats:
            # get output representations for probes in same category
            ps = make_output_representations(model, cs_scorer.probe_store.cat2probes[cat], prep)
            # compute divergences between exemplars within a category
            ws_cat = cs_scorer.calc_score(ps, ps,
                                          max_rows=configs.Eval.ws_max_rows)
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
                                   max_rows=configs.Eval.as_max_rows)

        performance.setdefault(f'as_n_{structure_name}', []).append(res)

    return performance


def update_di_performance(performance,
                          model: RNN,
                          prep: Prep,
                          structure2probe2cat: Dict[str, Dict[str, str]],
                          ):
    """
    compute average pairwise Euclidean ("ed") and cosine similarity ("cs") between probe representations.

    Cosine distance is defined as 1.0 minus the cosine similarity.

    """
    for structure_name in configs.Eval.structures:
        probe2cat = structure2probe2cat[structure_name]
        probe_store = ProbeStore(probe2cat)

        # get probe representations
        probe_reps_n = make_representations_without_context(model, probe_store.types, prep)

        # compute euclidean distance
        ed = euclidean_distances(probe_reps_n, probe_reps_n).mean()

        # calc average cosine similarity over off-diagonals (ignore 1s)
        sim = cosine_similarity(probe_reps_n)
        masked = np.ma.masked_where(np.eye(*sim.shape), sim)
        cs = masked.mean()

        performance.setdefault(f'ed_n_{structure_name}', []).append(ed)
        performance.setdefault(f'cs_n_{structure_name}', []).append(cs)

        # calc cosine similarity separately for same-category probes
        cc_total = 0
        for cat in probe_store.cats:
            # get probe representations
            probe_reps_n = make_representations_without_context(model, probe_store.cat2probes[cat], prep)
            sim = cosine_similarity(probe_reps_n)
            masked = np.ma.masked_where(np.eye(*sim.shape), sim)
            cc_cat = masked.mean()
            cc_total += cc_cat
        cc = cc_total / len(probe_store.cats)
        performance.setdefault(f'cc_n_{structure_name}', []).append(cc)

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

        probe_reps_n = make_representations_without_context(model, si_scorer.probe_store.types, prep)
        probe_reps_o = make_representations_with_context(model, si_scorer.probe_store.types, prep)
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

        probe_reps_n = make_representations_without_context(model, sd_scorer.probe_store.types, prep)
        probe_reps_o = make_representations_with_context(model, sd_scorer.probe_store.types, prep)
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

    do this by computing the output that results from a vector of zeros (origin),
    and computing its divergence from output distribution that corresponds to noun prototype
    """
    for structure_name in configs.Eval.structures:
        probe2cat = structure2probe2cat[structure_name]
        dp_scorer = DPScorer(probe2cat, prep.tokens, prep.types)

        # compute prototype at output layer
        p = dp_scorer._make_p(is_unconditional=False)[np.newaxis, :]

        # compute output that results at origin
        origin = np.zeros((1, model.hidden_size), dtype=np.float32)
        reshaped = origin[:, np.newaxis, :]  # embeddings must be in last (3rd dim)
        with torch.no_grad():
            embedded = torch.from_numpy(reshaped).cuda()
            encoded, _ = model.encode(embedded)
            last_encodings = torch.squeeze(encoded[:, -1])
            logits = model.project(last_encodings).cpu().numpy().astype(np.float64)
            q = softmax(logits[np.newaxis, :])  # softmax requires num dim = 2

        # compute score
        pi = np.asscalar(drv.divergence_jensenshannon_pmf(p, q))
        performance.setdefault(f'pi_n_{structure_name}', []).append(pi)

    return performance


def update_ep_performance(performance,
                          model: RNN,
                          prep: Prep,
                          structure2probe2cat: Dict[str, Dict[str, str]],
                          ):
    """
    compute average entropy of probe representations at output layer.

    also, compute entropy of representation of origin at output layer.
    """
    for structure_name in configs.Eval.structures:
        probe2cat = structure2probe2cat[structure_name]
        probe_store = ProbeStore(probe2cat)

        # compute representations of probes
        probe_reps_out = make_output_representations(model, probe_store.types, prep)

        # compute output that results at origin
        origin = np.zeros((1, model.hidden_size), dtype=np.float32)
        reshaped = origin[:, np.newaxis, :]  # embeddings must be in last (3rd dim)
        with torch.no_grad():
            embedded = torch.from_numpy(reshaped).cuda()
            encoded, _ = model.encode(embedded)
            last_encodings = torch.squeeze(encoded[:, -1])
            logits = model.project(last_encodings).cpu().numpy().astype(np.float64)
            origin_rep_out = softmax(logits[np.newaxis, :])  # softmax requires num dim = 2

        # compute entropy of probes
        ep = drv.entropy_pmf(probe_reps_out).mean()
        performance.setdefault(f'ep_n_{structure_name}', []).append(ep)

        # compute entropy of origin
        eo = drv.entropy_pmf(origin_rep_out).mean()
        performance.setdefault(f'eo_n_{structure_name}', []).append(eo)

    return performance


def update_fr_performance(performance,
                          model: RNN,
                          prep: Prep,
                          structure2probe2cat: Dict[str, Dict[str, str]],
                          ):
    """
    compute fragmentation = 1 - proportion of variance accounted for by first singular value of:

    1. probe representations at input (fragmentation at input, "fi"), and
    2. probe representations at output (fragmentation at output "fo")
    """
    for structure_name in configs.Eval.structures:
        probe2cat = structure2probe2cat[structure_name]
        probe_store = ProbeStore(probe2cat)

        # compute representations of probes
        probe_reps_n = make_representations_without_context(model, probe_store.types, prep)
        probe_reps_out = make_output_representations(model, probe_store.types, prep)

        # compute fragmentation at input
        u, s, vt = np.linalg.svd(probe_reps_n, compute_uv=True)
        fi = 1 - (s[0] / np.sum(s))
        performance.setdefault(f'fi_n_{structure_name}', []).append(fi)

        # compute fragmentation at output
        u, s, vt = np.linalg.svd(probe_reps_out, compute_uv=True)
        fo = 1 - (s[0] / np.sum(s))
        performance.setdefault(f'fo_n_{structure_name}', []).append(fo)

        # also compute condition number (ratio between first and last singular value)
        co = (s[0] / s[-1]) / np.sum(s)
        performance.setdefault(f'co_n_{structure_name}', []).append(co)

    return performance
