import torch
import numpy as np
from typing import List, Optional, Dict
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import CrossEntropyLoss
from pyitlib import discrete_random_variable as drv
from sklearn.metrics.cluster import silhouette_score, calinski_harabasz_score

from categoryeval.ba import BAScorer
from categoryeval.dp import DPScorer
from categoryeval.cs import CSScorer

from preppy import Prep

from childesrnnlm import configs
from childesrnnlm.s_dbw import S_Dbw
from childesrnnlm.rnn import RNN
from childesrnnlm.representation import softmax


def get_context2f(prep: Prep,
                  probes: List[str],
                  direction: str,
                  ) -> Dict[str, int]:
    """
    find the set of words that occur to the left/right of probes, and return them with their counts
    if a set is returned, this under-estimates the contribution of very frequent contexts,
     and over-estimates the contribution of infrequent contexts.

    the best strategy is  to return a dict with counts,
     and to use this to compute representations,
      and then to repeat each unique representation by the number of times the word was found.

    for sem-2021 probes and AO-CHILDES,
    there are approx 10K unique left contexts, while the full list has 200K contexts
    """

    offset = +1 if direction == 'r' else -1

    res = dict()
    for n, token in enumerate(prep.tokens[:-1]):  # do not remove elements from left side of tokens
        if prep.tokens[n+offset] in probes:
            res.setdefault(token, 0)
            res[token] += 1
    return res


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


def eval_ba_performance(representations: np.array,
                        probe2cat: Dict[str, str]
                        ):

    """
    balanced accuracy.

    Note:
        only works for probe representations (because category labels are required)
    """

    ba_scorer = BAScorer(probe2cat)
    probe_sims = cosine_similarity(representations)
    res = ba_scorer.calc_score(probe_sims, ba_scorer.gold_sims, 'ba', return_threshold=False)

    return res


def eval_si_performance(representations: np.array,
                        probe2cat: Dict[str, str]
                        ):
    """
    silhouette score.

    how well do probe representations cluster with representations of probes in the same class?
    """

    categories = sorted(set(probe2cat.values()))
    category_labels = [categories.index(probe2cat[p]) for p in probe2cat]

    # compute silhouette score
    res = silhouette_score(representations, category_labels, metric='cosine')

    return res


def eval_sd_performance(representations: np.array,
                        probe2cat: Dict[str, str]
                        ):
    """
    S-Dbw score.

    how well do probe representations cluster with representations of probes in the same class?

    using code from https://github.com/alashkov83/S_Dbw
    """

    categories = sorted(set(probe2cat.values()))
    category_labels = [categories.index(probe2cat[p]) for p in probe2cat]

    res = S_Dbw(representations,
                category_labels,
                centers_id=None,
                method='Tong',
                alg_noise='bind',
                centr='mean',
                nearest_centr=True,
                metric='cosine')

    return res


def eval_ma_performance(representations: np.array,
                        ):
    """
    magnitude of representations.
    """

    ma = np.linalg.norm(representations, axis=1).mean()  # computes magnitude for each vector, then mean

    return ma


def eval_pr1_performance(representations: np.array,
                         types_eval: List[str],
                         prep: Prep,
                         model: RNN,
                         ):
    """
    divergence between actual and theoretical prototype.

    the "actual prototype" is computed by averaging the representations of probes on their first singular dimension.
    next, the output of the model is computed, and compared to the "theoretical prototype",
    a probability distribution constructed by averaging over all probes' next-word probability distributions.

    """
    dp_scorer = DPScorer(types_eval, prep.tokens, prep.types)

    # compute average projection on first singular dimension
    u, s, vt = np.linalg.svd(representations, compute_uv=True)
    projections = s[0] * u[:, 0].reshape(-1, 1) @ vt[0, :].reshape(1, -1)
    prototype_actual = projections.mean(axis=0)

    # compute output that results from actual-prototype
    reshaped = prototype_actual[np.newaxis, np.newaxis, :].astype(np.float32)  # embeddings must be in 3rd dim
    with torch.no_grad():
        embedded = torch.from_numpy(reshaped).cuda()
        encoded, _ = model.encode(embedded)
        last_encodings = torch.squeeze(encoded[:, -1])
        logits = model.project(last_encodings).cpu().numpy().astype(np.float64)
        prototype_actual_out = softmax(logits[np.newaxis, :])  # softmax requires num dim = 2

    res = dp_scorer.score(prototype_actual_out)

    return res


def eval_pr2_performance(representations: np.array,
                         types_eval: List[str],
                         prep: Prep,
                         ):
    """
    divergence between exemplars and theoretical prototype.

    a home-made quantity that is proportional to the distance of all probe representations
    to a location in representational space that can be thought of as the probe (typically nouns) "prototype",
    obtained by leveraging all of the co-occurrence data in a corpus.

    """

    dp_scorer = DPScorer(types_eval, prep.tokens, prep.types)
    res = dp_scorer.score(representations, return_mean=True, metric='js')

    return res


def eval_pd_performance(representations: np.array,
                        ):
    """
    pairwise divergences.

    """
    cs_scorer = CSScorer()
    res = cs_scorer.calc_score(representations,
                               representations,
                               max_rows=configs.Eval.as_max_rows)

    return res


def eval_cs_performance(representations: np.array,
                        probe2cat: Dict[str, str]
                        ):
    """
    cosine similarity.

    Cosine distance is defined as 1.0 minus the cosine similarity.

    """

    # calc average cosine similarity off-diagonals (ignore 1s)
    sim = cosine_similarity(representations)
    masked = np.ma.masked_where(np.eye(*sim.shape), sim)
    cs = masked.mean()

    # calc cosine similarity separately for same-category probes
    categories = sorted(set(probe2cat.values()))
    probes = sorted(set(probe2cat.keys()))
    cc_total = 0
    for cat in categories:
        # get probe representations
        idx = [n for n, p in enumerate(probes) if probe2cat[p] == cat]
        representations_cat = representations[idx]
        sim = cosine_similarity(representations_cat)
        masked = np.ma.masked_where(np.eye(*sim.shape), sim)
        cc_cat = masked.mean()
        cc_total += cc_cat
    cc = cc_total / len(categories)

    return cs, cc


def eval_op_performance(model: RNN,
                        prep: Prep,
                        types_eval: List[str]
                        ):
    """
    divergence of origin from teh prototype.

    do this by computing the output that results from a vector of zeros (origin),
    and computing its divergence from output distribution that corresponds to the prototype of types_eval
    """
    dp_scorer = DPScorer(types_eval, prep.tokens, prep.types)

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

    res = np.asscalar(drv.divergence_jensenshannon_pmf(p, q))

    return res


def eval_en_performance(representations: np.array,
                        model: RNN,
                        ):
    """
    entropy of probe representations at output layer and at origin.
    """

    # compute output that results at origin
    origin = np.zeros((1, model.hidden_size), dtype=np.float32)
    reshaped = origin[:, np.newaxis, :]  # embeddings must be in last (3rd dim)
    with torch.no_grad():
        embedded = torch.from_numpy(reshaped).cuda()
        encoded, _ = model.encode(embedded)
        last_encodings = torch.squeeze(encoded[:, -1])
        logits = model.project(last_encodings).cpu().numpy().astype(np.float64)
        representation_origin = softmax(logits[np.newaxis, :])  # softmax requires num dim = 2

    # compute entropy of representations
    ep = drv.entropy_pmf(representations).mean()

    # compute entropy of origin
    eo = drv.entropy_pmf(representation_origin).mean()

    return ep, eo


def eval_fr_performance(representations: np.array,
                        type_eval2f: Optional[Dict[str, int]]
                        ):
    """
    fragmentation = 1 - proportion of variance accounted for by first singular value



    # TODO this function does not replicate results generated by previous function (8/13):

    def update_cf_performance(performance,
                          model: RNN,
                          prep: Prep,
                          structure2probe2cat: Dict[str, Dict[str, str]],
                          ):

        for structure_name in configs.Eval.structures:
            probe2cat = structure2probe2cat[structure_name]
            probe_store = ProbeStore(probe2cat)

            # get representations (frequency is considered by collecting all instances of context words, rather than types)
            left_context2f = get_left_context2f(prep, probe_store.types)
            left_contexts = list(left_context2f.keys())
            print(f'Found {len(left_contexts):,} unique left context words')
            left_contexts_reps_out = make_output_representations(model, left_contexts, prep)

            # repeat each representation by the number of times it occurs to the left of probe words.
            # note: this considers the frequency of left contexts
            repeats = [left_context2f[lc] for lc in left_contexts]
            left_contexts_reps_out_weighted = np.repeat(left_contexts_reps_out, repeats, axis=0)

            # score
            s = np.linalg.svd(left_contexts_reps_out_weighted, compute_uv=False)
            cf = 1 - (s[0] / np.sum(s))
            performance.setdefault(f'cf_n_{structure_name}', []).append(cf)

        return performance

    """

    # repeat each context-word representation by the number of times it occurs with a probe word.
    # note: this considers the frequency of context words
    if configs.Eval.frequency_weighting and type_eval2f:
        repeats = [f for t, f in type_eval2f.items()]
        assert len(repeats) == len(representations)
        mat = np.repeat(representations, repeats, axis=0)
        assert mat.shape[1] == representations.shape[1]
    else:
        mat = representations

    # compute fragmentation
    s = np.linalg.svd(mat, compute_uv=False)
    fr = 1 - (s[0] / np.sum(s))

    # also compute condition number (ratio between first and last singular value)
    co = (s[0] / s[-1]) / np.sum(s)

    return fr, co
