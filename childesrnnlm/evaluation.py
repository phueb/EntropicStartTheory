import torch
import numpy as np
from typing import List, Optional, Dict
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
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
from childesrnnlm.representation import softmax, make_inp_representations_without_context


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
    # offset should determine where to look for a probe.
    # specifically, when context == 'l', we check for probes to the right, that is at location  n + 1
    offset = +1 if direction == 'l' else -1

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
        last_output = torch.squeeze(encoded[:, -1])
        logits = model.project(last_output).cpu().numpy().astype(np.float64)
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
    pairwise divergence.

    """
    cs_scorer = CSScorer()
    res = cs_scorer.calc_score(representations,
                               representations,
                               max_rows=configs.Eval.as_max_rows)

    return res


def eval_pe_performance(representations: np.array,
                        ):
    """
    pairwise euclidean distance.

    """
    res = np.asscalar(euclidean_distances(representations).mean())

    return res


def eval_cs_performance(representations: np.array,
                        ):
    """
    cosine similarity.

    Cosine distance is defined as 1.0 minus the cosine similarity.

    """

    # calc average cosine similarity off-diagonals (ignore 1s)
    sim = cosine_similarity(representations)
    masked = np.ma.masked_where(np.eye(*sim.shape), sim)
    res = masked.mean()

    return res


def eval_cc_performance(representations: np.array,
                        probe2cat: Dict[str, str]
                        ):
    """
    cosine similarity averaged within each category
    """

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
    res = cc_total / len(categories)

    return res


def eval_op_performance(model: RNN,
                        prep: Prep,
                        types_eval: List[str]
                        ):
    """
    divergence of origin from the prototype.

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
        last_output = torch.squeeze(encoded[:, -1])
        logits = model.project(last_output).cpu().numpy().astype(np.float64)
        q = softmax(logits[np.newaxis, :])  # softmax requires num dim = 2

    res = np.asscalar(drv.divergence_jensenshannon_pmf(p, q))

    return res


def eval_en_performance(representations: np.array,
                        ):
    """
    entropy of probe representations at output layer and at origin.
    """

    res = drv.entropy_pmf(representations).mean()

    return res


def eval_eo_performance(model: RNN,
                        ):
    """
    entropy of representation at output of origin.
    """

    # compute output that results at origin
    origin = np.zeros((1, model.hidden_size), dtype=np.float32)
    reshaped = origin[:, np.newaxis, :]  # embeddings must be in last (3rd dim)
    with torch.no_grad():
        embedded = torch.from_numpy(reshaped).cuda()
        encoded, _ = model.encode(embedded)
        last_output = torch.squeeze(encoded[:, -1])
        logits = model.project(last_output).cpu().numpy().astype(np.float64)
        representation_origin = softmax(logits[np.newaxis, :])  # softmax requires num dim = 2

    res = drv.entropy_pmf(representation_origin).mean()

    return res


def eval_fr_performance(representations: np.array,
                        type_eval2f: Optional[Dict[str, int]]
                        ):
    """
    fragmentation = 1 - proportion of variance accounted for by first singular value
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
    res = 1 - (s[0] / np.sum(s))

    return res


def eval_cd_performance(model: RNN,
                        prep: Prep,
                        types_eval: List[str],
                        max_num_exemplars: int = 128,
                        ):
    """
    context divergence.
    divergence between outputs produce by the same probe + different contexts and contextualized probe prototype.
    """

    all_windows = prep.reordered_windows

    kls = []
    w_ids = [prep.token2id[w] for w in types_eval]
    for n, token_id in enumerate(w_ids):
        bool_idx = np.isin(all_windows[:, -2], token_id)

        x = all_windows[bool_idx][:, :-1]

        # skip if probe occurs once only - divergence = 0 in such cases
        if len(x) == 1:
            continue

        # feed-forward
        inputs = torch.LongTensor(x).cuda()
        logits = model(inputs)['logits'].detach().cpu().numpy()
        last_output = model(inputs)['last_output']

        # make q
        q = softmax(logits)

        # make p
        prototype = torch.mean(last_output, dim=0, keepdim=True)  # [1, hidden_size]
        logits = model.project(prototype).detach().cpu().numpy()  # 2D is preserved
        p = softmax(logits)

        # need to cast from float32 to float64 to avoid very slow check for NaNs in drv.divergence_jensenshannon_pmf
        p = p.astype(np.float64)
        q = q.astype(np.float64)

        # print(p.shape, q.shape, flush=True)

        kl_i = drv.divergence_kullbackleibler_pmf(p, q, cartesian_product=True).mean()
        kls.append(kl_i)

    res = np.mean(kls)

    return res


def eval_ds_performance(model: RNN,
                        prep: Prep,
                        types_eval: List[str],
                        max_num_exemplars: int = 128,
                        ):
    """
    divergence from superordinate.
    divergence between outputs produced by probe + different context and contextualized superordinate prototype

    Notes:
        p and q contain rows of probability distributions.
        because each p is paired with one q, the total number of comparisons = len(p) = len(q).
        p contains probabilities given a prototype in last position of the input sequence..
        q contains probabilities given an actual probe in last position of the input sequence.
    """

    all_windows = prep.reordered_windows

    embeddings_probe = make_inp_representations_without_context(model, types_eval, prep)
    superordinate_prototype = embeddings_probe.mean(axis=0)

    probes = types_eval

    kls = []
    for type_eval in types_eval:
        token_id = prep.token2id[type_eval]
        bool_idx = np.isin(all_windows[:, -2], token_id)

        # exclude y, and also exclude probe - equivalent to context_type='m'
        x = all_windows[bool_idx][:, :-2]

        # skip if probe occurs once only - divergence = 0 in such cases
        if len(x) == 1:
            continue

        # feed-forward without probe
        inputs = torch.LongTensor(x).cuda()
        model_output_dict = model(inputs)

        # make p and q
        if model.flavor == 'rnn':

            last_output = model_output_dict['last_output']

            # make p (by adding the prototype at the last step)
            inputs_p = np.repeat(superordinate_prototype[np.newaxis, :],
                                 len(x), 0)
            hiddens_p, _ = model.encode(torch.from_numpy(inputs_p[:, np.newaxis, :].astype(np.float32)).cuda(),
                                        torch.unsqueeze(last_output, 0))
            logits_p = model.project(torch.squeeze(hiddens_p[:, -1])).detach().cpu().numpy()  # 2D preserved
            p = softmax(logits_p)  # [num exemplars, hidden_size]

            # make q (by adding the probe to the last step)
            probe_id = probes.index(type_eval)
            inputs_q = np.repeat(embeddings_probe[probe_id][np.newaxis, :],
                                 len(inputs), 0)
            hiddens_q, _ = model.encode(torch.from_numpy(inputs_q[:, np.newaxis, :].astype(np.float32)).cuda(),
                                        torch.unsqueeze(last_output, 0))
            logits_q = model.project(torch.squeeze(hiddens_q[:, -1])).detach().cpu().numpy()  # 2D preserved
            q = softmax(logits_q)  # [num exemplars, hidden_size]

        # TODO test
        elif model.flavor == 'lstm':   # we need both h_n and c_n, not just h_n in the case of the RNN

            h_previous = model_output_dict['h_n']
            c_previous = model_output_dict['c_n']

            # make p (by adding the prototype at the last step)
            inputs_p = np.repeat(superordinate_prototype[np.newaxis, :],
                                 len(x), 0)
            hiddens_p, _ = model.encode(torch.from_numpy(inputs_p[:, np.newaxis, :].astype(np.float32)).cuda(),
                                        (h_previous, c_previous))
            logits_p = model.project(torch.squeeze(hiddens_p[:, -1])).detach().cpu().numpy()  # 2D preserved
            p = softmax(logits_p)  # [num exemplars, hidden_size]

            # make q (by adding the probe to the last step)
            probe_id = probes.index(type_eval)
            inputs_q = np.repeat(embeddings_probe[probe_id][np.newaxis, :],
                                 len(inputs), 0)
            hiddens_q, _ = model.encode(torch.from_numpy(inputs_q[:, np.newaxis, :].astype(np.float32)).cuda(),
                                        (h_previous, c_previous))
            logits_q = model.project(torch.squeeze(hiddens_q[:, -1])).detach().cpu().numpy()  # 2D preserved
            q = softmax(logits_q)  # [num exemplars, hidden_size]
        else:
            raise AttributeError(f'Invalid arg to "flavor"')

        # need to cast from float32 to float64 to avoid very slow check for NaNs in drv.divergence_jensenshannon_pmf
        p = p.astype(np.float64)
        q = q.astype(np.float64)

        # print(p.shape, q.shape, flush=True)  # should be the same size - we evaluate each row with one paired row

        kl_i = drv.divergence_kullbackleibler_pmf(p, q, cartesian_product=False).mean()
        kls.append(kl_i)

    res = np.mean(kls)

    return res
