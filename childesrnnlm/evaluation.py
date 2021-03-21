import pyprind
import torch
import numpy as np

from typing import List, Union, Dict
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import CrossEntropyLoss

from categoryeval.ba import BAScorer
from categoryeval.dp import DPScorer
from categoryeval.cs import CSScorer
from categoryeval.si import SIScorer
from categoryeval.sd import SDScorer

from preppy import FlexiblePrep

from childesrnnlm import configs
from childesrnnlm.rnn import RNN
from childesrnnlm.representation import make_representations_without_context
from childesrnnlm.representation import make_representations_with_context
from childesrnnlm.representation import make_output_representation
from childesrnnlm.io import load_probe2cat


def calc_perplexity(model: RNN,
                    criterion: CrossEntropyLoss,
                    prep: FlexiblePrep,
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
                          prep: FlexiblePrep,
                          ):
    if not configs.Eval.train_pp:
        if configs.Eval.min_num_test_tokens > 0:
            test_pp = calc_perplexity(model, criterion, prep, is_test=True)
            performance['test_pp'].append(test_pp)
    else:
        if configs.Eval.min_num_test_tokens > 0:
            train_pp = calc_perplexity(model, criterion, prep, is_test=False)
            test_pp = calc_perplexity(model, criterion, prep, is_test=True)
            performance['train_pp'].append(train_pp)
            performance['test_pp'].append(test_pp)
    return performance


def update_ba_performance(performance,
                          model: RNN,
                          prep: FlexiblePrep,
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

        if configs.Eval.ba_n:
            performance.setdefault(f'ba_n_{structure_name}', []).append(
                ba_scorer.calc_score(probe_sims_n, probe_store.gold_sims, 'ba'))
        if configs.Eval.ba_o:
            performance.setdefault(f'ba_o_{structure_name}', []).append(
                ba_scorer.calc_score(probe_sims_o, probe_store.gold_sims, 'ba'))

    return performance


def update_dp_performance(performance,
                          model: RNN,
                          prep: FlexiblePrep,
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
        qs = make_output_representation(model, probes, prep)

        # check predictions
        max_ids = np.argsort(qs.mean(axis=0))
        print(f'{structure_name} predict:', [prep.types[i] for i in max_ids[-10:]])

        # dp
        performance.setdefault(f'dp_{structure_name}_js', []).append(dp_scorer.calc_dp(qs, structure_name, metric='js'))

    return performance


def update_cs_performance(performance,
                          model: RNN,
                          prep: FlexiblePrep,
                          structure2probe2cat: Dict[str, Dict[str, str]],
                          ):
    """
    compute category-spread.
    a home-made quantity that is proportional to the spread between probe representations in the same category
    """
    exemplars_list = []
    for structure_name in configs.Eval.structures:
        probe2cat = structure2probe2cat[structure_name]
        cs_scorer = CSScorer(probe2cat)

        # collect exemplars
        for cat in cs_scorer.probe_store.cats:
            exemplars = make_output_representation(model, cs_scorer.probe_store.cat2probes[cat], prep)
            exemplars_list.append(exemplars)
        ps = np.vstack(exemplars_list)

        # compute divergences between exemplars within a category (not prototypes - produces noisy plot)
        dp = cs_scorer.calc_cs(ps, ps, metric='js', max_rows=configs.Eval.cs_max_rows)
        performance.setdefault(f'cs_{structure_name}_js', []).append(dp)

    return performance


def update_si_performance(performance,
                          model: RNN,
                          prep: FlexiblePrep,
                          structure2probe2cat: Dict[str, Dict[str, str]],
                          ):
    """
    compute silhouette scores.
    how well do probe representations cluster with representations of probes in the same class?
    """
    for structure_name in configs.Eval.structures:
        probe2cat = structure2probe2cat[structure_name]
        si_scorer = SIScorer(probe2cat)

        probe_store = si_scorer.probe_store
        probe_token_ids = [prep.token2id[token] for token in probe_store.types]

        probe_reps_n = make_representations_without_context(model, probe_token_ids)
        probe_reps_o = make_representations_with_context(model, probe_token_ids, prep)
        cat_ids = [probe_store.cat2id[probe_store.probe2cat[p]] for p in probe_store.types]

        # compute silhouette score
        if configs.Eval.si_n:
            performance.setdefault(f'si_n_{structure_name}', []).append(
                si_scorer.calc_si(probe_reps_n, cat_ids))
        if configs.Eval.si_o:
            performance.setdefault(f'si_o_{structure_name}', []).append(
                si_scorer.calc_si(probe_reps_o, cat_ids))

    return performance


def update_sd_performance(performance,
                          model: RNN,
                          prep: FlexiblePrep,
                          structure2probe2cat: Dict[str, Dict[str, str]],
                          ):
    """
    compute S-Dbw score.
    how well do probe representations cluster with representations of probes in the same class?
    """
    for structure_name in configs.Eval.structures:
        probe2cat = structure2probe2cat[structure_name]
        sd_scorer = SDScorer(probe2cat)

        probe_store = sd_scorer.probe_store
        probe_token_ids = [prep.token2id[token] for token in probe_store.types]

        probe_reps_n = make_representations_without_context(model, probe_token_ids)
        probe_reps_o = make_representations_with_context(model, probe_token_ids, prep)
        cat_ids = [probe_store.cat2id[probe_store.probe2cat[p]] for p in probe_store.types]

        # compute score
        if configs.Eval.sd_n:
            performance.setdefault(f'sd_n_{structure_name}', []).append(
                sd_scorer.calc_sd(probe_reps_n, cat_ids))
        if configs.Eval.sd_o:
            performance.setdefault(f'sd_o_{structure_name}', []).append(
                sd_scorer.calc_sd(probe_reps_o, cat_ids))

    return performance


def get_weights(model):
    ih = model.rnn.weight_ih_l  # [hidden_size, input_size]
    hh = model.rnn.weight_hh_l  # [hidden_size, hidden_size]
    return {'ih': ih, 'hh': hh}