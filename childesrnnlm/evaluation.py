import pyprind
import torch
import numpy as np

from typing import List, Union, Dict
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import CrossEntropyLoss

from categoryeval.ba import BAScorer


from preppy import Prep

from childesrnnlm import configs
from childesrnnlm.rnn import RNN
from childesrnnlm.representation import make_representations_without_context
from childesrnnlm.representation import make_representations_with_context


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


def update_ba_performance(performance,
                          model: RNN,
                          prep: Prep,
                          structure2probe2cat: Dict[str, Dict[str, str]],
                          ):
    for structure_name in structure2probe2cat:
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


def get_weights(model):
    ih = model.rnn.weight_ih_l  # [hidden_size, input_size]
    hh = model.rnn.weight_hh_l  # [hidden_size, hidden_size]
    return {'ih': ih, 'hh': hh}