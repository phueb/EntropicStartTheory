import pyprind
import torch
import numpy as np

from typing import List, Union, Dict
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import CrossEntropyLoss

from categoryeval.ra import RAScorer
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


def update_ra_performance(performance,
                          model: RNN,
                          prep: Prep,
                          structure2probe2cat: Dict[str, Dict[str, str]],
                          offset_percent: float = 0.01,
                          ):
    """
    compute raggedness of input-output mapping.
    """
    for structure_name in configs.Eval.structures:
        probe2cat = structure2probe2cat[structure_name]
        ra_scorer = RAScorer(probe2cat)

        # get probe representations
        probe_token_ids = [prep.token2id[token] for token in ra_scorer.probe_store.types]
        probe_reps_inp = make_representations_without_context(model, probe_token_ids)
        probe_reps_out = make_output_representations(model, ra_scorer.probe_store.types, prep)

        ra_total = 0
        for probe_rep_inp, probe_rep_out in zip(probe_reps_inp, probe_reps_out):

            probe_rep_inp_tiled = np.tile(probe_rep_inp, (len(probe_rep_inp), 1))
            probe_rep_out_tiled = np.tile(probe_rep_out, (len(probe_rep_inp), 1))

            # get vectors representing locations close to probe representations.
            # note: these locations are created by adding offset to a single dimension
            offset_magnitude = offset_percent * np.max(probe_rep_inp_tiled)
            offset = np.eye(probe_rep_inp_tiled.shape[0], probe_rep_inp_tiled.shape[1]) * offset_magnitude
            nearby_reps_inp = probe_rep_inp_tiled + offset

            # get outputs for nearby_reps
            inputs = torch.cuda.LongTensor(nearby_reps_inp)
            logits = model(inputs)['logits'].detach().cpu().numpy()
            nearby_reps_out = softmax(logits)

            # calc score for a single probe and collect
            ra_probe = ra_scorer.calc_score(probe_rep_out_tiled, nearby_reps_out,
                                            metric=configs.Eval.ra_metric)  # TODO fine-tune this function
            ra_total += ra_probe

        ra = ra_total / len(ra_scorer.probe_store.types)
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

        performance.setdefault(f'ba_n_{structure_name}', []).append(
            ba_scorer.calc_score(probe_sims_n, probe_store.gold_sims, 'ba'))

        performance.setdefault(f'ba_o_{structure_name}', []).append(
            ba_scorer.calc_score(probe_sims_o, probe_store.gold_sims, 'ba'))

    return performance


def update_dp_performance(performance,
                          model: RNN,
                          prep: Prep,
                          structure2probe2cat: Dict[str, Dict[str, str]],
                          ):
    """
    calculate distance-to-prototype (aka dp):
    a home-made quantity that is proportional to the distance of same-category probe representations
    to a location in representational space that can be thought of as the category's "prototype",
    obtained by leveraging all of the co-occurrence data in a corpus.
    """
    for structure_name in configs.Eval.structures:
        probe2cat = structure2probe2cat[structure_name]
        dp_scorer = DPScorer(probe2cat, prep.tokens)

        # collect dp for probes who tend to occur most frequently in some part of corpus
        probes = dp_scorer.probe_store.types
        qs = make_output_representations(model, probes, prep)

        # check predictions
        max_ids = np.argsort(qs.mean(axis=0))
        print(f'{structure_name} predict:', [prep.types[i] for i in max_ids[-10:]])

        # dp
        dp = dp_scorer.calc_dp(qs, return_mean=True, metric='js')
        performance.setdefault(f'dp_n_{structure_name}', []).append(dp)

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
            ws_cat = cs_scorer.calc_cs(ps, ps,
                                       metric=configs.Eval.cs_metric,
                                       max_rows=configs.Eval.cs_max_rows)
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

        # get probe representations
        probe_reps_out = make_output_representations(model, cs_scorer.probe_store.types, prep)

        as_total = 0
        for probe, probe_rep_out in zip(cs_scorer.probe_store.types, probe_reps_out):

            # compute divergences between one probe representation and all other probe representations
            as_probe = cs_scorer.calc_cs(probe_rep_out[np.newaxis, :], probe_reps_out,
                                         metric=configs.Eval.cs_metric,
                                         max_rows=configs.Eval.cs_max_rows)
            print(f'across-category spread for probe={probe:<18} ={as_probe:.4f}', flush=True)
            as_total += as_probe

        as_ = as_total / len(cs_scorer.probe_store.cats)
        performance.setdefault(f'as_n_{structure_name}', []).append(as_)

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


def get_weights(model):
    ih = model.rnn.weight_ih_l  # [hidden_size, input_size]
    hh = model.rnn.weight_hh_l  # [hidden_size, hidden_size]
    return {'ih': ih, 'hh': hh}