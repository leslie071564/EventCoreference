from collections import defaultdict
import itertools
import operator
import re
import os
import subprocess
import tempfile
import warnings
import time
import functools
import array
import signal

from scipy import sparse
from scipy.sparse import csgraph

import numpy as np
from munkres import linear_assignment

values = dict.values
keys = dict.keys


class TimeoutError(Exception):
    pass


class timeout:
    def __init__(self, seconds=1):
        self.seconds = seconds

    def handle_timeout(self, signum, frame):
        raise TimeoutError()

    def __enter__(self):
        try:
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)
        except Exception:
            # I can't find documentation of what happens if executed off Unix
            pass

    def __exit__(self, type, value, traceback):
        try:
            signal.alarm(0)
        except Exception:
            # I can't find documentation of what happens if executed off Unix
            pass

######## Utilities ########

def mapping_to_sets(mapping):
    """
    Input: {cluster_item: cluster_name} dictionary
    Output: {cluster_name: set([cluster_items])} dictionary
    """
    s = defaultdict(set)
    for m, k in mapping.items():
        s[k].add(m)
    s.default_factory = None  # disable defaulting
    return s


def sets_to_mapping(s):
    """
    Input: {cluster_name: set([cluster_items])} dictionary
    Output: {cluster_item: cluster_name} dictionary
    """
    return {m: k for k, ms in s.items() for m in ms}


def _f1(a, b):
    if a + b:
        return 2 * a * b / (a + b)
    return 0.


def _prf(p_num, p_den, r_num, r_den):
    p = p_num / p_den if p_den > 0 else 0.
    r = r_num / r_den if r_den > 0 else 0.
    return p, r, _f1(p, r)


def twinless_adjustment(true, pred):
    """Adjusts predictions for differences in mentions

    Following Cai and Strube's (SIGDIAL'10) `sys` variants on B-cubed and CEAF.
    This produces a different true, pred pair for each of precision and recall
    calculation.

    Thus for precision:
        * twinless true mentions -> pred singletons
        * twinless pred singletons -> discard
        * twinless pred non-singletons -> true singletons
    For recall:
        * twinless true -> pred singletons
        * twinless pred -> discard

    Returns : p_true, p_pred, r_true, r_pred
    """
    true_mapping = sets_to_mapping(true)
    pred_mapping = sets_to_mapping(pred)

    # common: twinless true -> pred singletons
    twinless_true = set(true_mapping) - set(pred_mapping)
    for i, mention in enumerate(twinless_true):
        pred_mapping[mention] = ('twinless_true', i)

    # recall: twinless pred -> discard
    r_pred = mapping_to_sets({m: k
                              for m, k in pred_mapping.items()
                              if m in true_mapping})

    # precision: twinless pred singletons -> discard; non-singletons -> true
    for i, (m, k) in enumerate(list(pred_mapping.items())):
        if m in true_mapping:
            continue
        if len(pred[k]) > 1:
            true_mapping[m] = ('twinless_pred', i)
        else:
            del pred_mapping[m]

    p_true = mapping_to_sets(true_mapping)
    p_pred = mapping_to_sets(pred_mapping)

    return p_true, p_pred, true, r_pred


def sets_to_matrices(true, pred):
    vocabulary = defaultdict(None)
    vocabulary.default_factory = vocabulary.__len__
    true_indptr = array.array('i', [0])
    true_indices = array.array('i')
    for true_cluster in values(true):
        for item in true_cluster:
            true_indices.append(vocabulary[item])
        true_indptr.append(len(vocabulary))

    pred_indptr = array.array('i', [0])
    pred_indices = array.array('i')
    for pred_cluster in values(pred):
        for item in pred_cluster:
            pred_indices.append(vocabulary[item])
        pred_indptr.append(len(pred_indices))

    true_data = np.ones(len(true_indices), dtype=int)
    true_matrix = sparse.csr_matrix((true_data, true_indices, true_indptr),
                                    shape=(len(true), len(vocabulary)))
    pred_data = np.ones(len(pred_indices), dtype=int)
    pred_matrix = sparse.csr_matrix((pred_data, pred_indices, pred_indptr),
                                    shape=(len(pred), len(vocabulary)))

    return true_matrix, pred_matrix, vocabulary


######## Cluster comparison ########

def dice(a, b):
    """

    "Entity-based" measure in CoNLL; #4 in CEAF paper
    """
    if a and b:
        return (2 * len(a & b)) / (len(a) + len(b))
    return 0.


def _vectorized_dice(true_matrix, pred_matrix):
    overlap = _vectorized_overlap(true_matrix, pred_matrix).astype(float)

    # The following should be no-ops
    assert overlap.format == true_matrix.format == pred_matrix.format == 'csr'

    true_sizes = np.diff(true_matrix.indptr)
    pred_sizes = np.diff(pred_matrix.indptr)

    denom = np.repeat(true_sizes, np.diff(overlap.indptr))
    denom += pred_sizes.take(overlap.indices)
    overlap.data *= 2 / denom

    return overlap

dice.vectorized = _vectorized_dice


def overlap(a, b):
    """Intersection of sets

    "Mention-based" measure in CoNLL; #3 in CEAF paper
    """
    return len(a & b)


def _vectorized_overlap(true_matrix, pred_matrix):
    return true_matrix * pred_matrix.T

overlap.vectorized = _vectorized_overlap


######## Coreference metrics ########


def _disjoint_max_assignment(similarities):
    global sparse

    # form n*n adjacency matrix
    where_true, where_pred = similarities.nonzero()
    where_pred = where_pred + similarities.shape[0]
    n = sum(similarities.shape)
    A = sparse.coo_matrix((np.ones(len(where_true)), (where_true, where_pred)),
                          shape=(n, n))
    try:
        n_components, components = csgraph.connected_components(A, directed=False)
    except (AttributeError, TypeError):
        warnings.warn('Could not use scipy.sparse.csgraph.connected_components.'
                      'Please update your scipy installation. '
                      'Calculating max-score assignment the slow way.')
        # HACK!
        sparse = None
        return _disjoint_max_assignment(similarities)

    if hasattr(similarities, 'toarray'):
        # faster to work in dense
        similarities = similarities.toarray()
    total = 0
    for i in range(n_components):
        mask = components == i
        component_true = np.flatnonzero(mask[:similarities.shape[0]])
        component_pred = np.flatnonzero(mask[similarities.shape[0]:])
        component_sim = similarities[component_true, :][:, component_pred]
        if 0 in component_sim.shape:
            pass
        if component_sim.shape == (1, 1):
            total += component_sim[0, 0]
        else:
            indices = linear_assignment(-component_sim)
            total += component_sim[indices[:, 0], indices[:, 1]].sum()
    return total


def ceaf(true, pred, similarity=dice):
    "Luo (2005). On coreference resolution performance metrics. In EMNLP."
    try:
        with timeout(900):
            true, pred, _ = sets_to_matrices(true, pred)
            X = similarity.vectorized(true, pred)
            p_num = r_num = _disjoint_max_assignment(X)
            r_den = similarity.vectorized(true, true).sum()
            p_den = similarity.vectorized(pred, pred).sum()
    except TimeoutError:
        warnings.warn('timeout for CEAF!')
        return 0, 0, 0, 0

    return p_num, p_den, r_num, r_den


def cs_ceaf(true, pred, similarity=dice):
    """CEAF with twinless adjustment from Cai and Strube (2010)"""
    p_true, p_pred, r_true, r_pred = twinless_adjustment(true, pred)
    # XXX: there is probably a better way to do this
    p_num, p_den, _, _ = ceaf(p_true, p_pred, similarity)
    _, _, r_num, r_den = ceaf(r_true, r_pred, similarity)
    return p_num, p_den, r_num, r_den


def mention_ceaf(true, pred):
    "Luo (2005) phi_3"
    return ceaf(true, pred, similarity=overlap)


def entity_ceaf(true, pred):
    "Luo (2005) phi_4"
    return ceaf(true, pred, similarity=dice)


def mention_cs_ceaf(true, pred):
    return cs_ceaf(true, pred, similarity=overlap)


def entity_cs_ceaf(true, pred):
    return cs_ceaf(true, pred, similarity=dice)


def _b_cubed(A, B, EMPTY=frozenset([])):
    A_mapping = sets_to_mapping(A)
    B_mapping = sets_to_mapping(B)
    res = 0.
    for m, k in A_mapping.items():
        A_cluster = A.get(k, EMPTY)
        res += len(A_cluster & B.get(B_mapping.get(m), EMPTY)) / len(A_cluster)
    return res, len(A_mapping)


def b_cubed(true, pred):
    """
    Bagga and Baldwin (1998). Algorithms for scoring coreference chains.
    In LREC Linguistic Coreference Workshop.

    TODO: tests
    """
    p_num, p_den = _b_cubed(pred, true)
    r_num, r_den = _b_cubed(true, pred)
    return p_num, p_den, r_num, r_den


def cs_b_cubed(true, pred):
    """b_cubed with twinless adjustment from Cai and Strube (2010)"""
    p_true, p_pred, r_true, r_pred = twinless_adjustment(true, pred)
    p_num, p_den = _b_cubed(p_pred, p_true)
    r_num, r_den = _b_cubed(r_true, r_pred)
    return p_num, p_den, r_num, r_den


def _positive_pairs(C):
    "Return pairs of instances across all clusters in C"
    return frozenset(itertools.chain.from_iterable(
        itertools.combinations(sorted(c), 2) for c in C))


def _negative_pairs(C):
    return frozenset(tuple(sorted(item_pair))
                     for cluster_pair in itertools.combinations(C, 2)
                     for item_pair in itertools.product(*cluster_pair))

def _pairwise(true, pred):
    """Return numerators and denominators for precision and recall,
    as well as size of symmetric difference, used in negative pairwise."""
    p_num = r_num = len(true & pred)
    p_den = len(pred)
    r_den = len(true)
    return p_num, p_den, r_num, r_den


def pairwise(true, pred):
    """Return p_num, p_den, r_num, r_den over item pairs

    As used in calcualting BLANC (see Luo, Pradhan, Recasens and Hovy (2014).

    >>> pairwise({1: {'a', 'b', 'c'}, 2: {'d'}},
    ...         {1: {'b', 'c'}, 2: {'d', 'e'}})
    (1, 2, 1, 3)
    """
    # Slow in some cases, perhaps, but much less memory consumed
    return pairwise_slow(true, pred)
    return _pairwise(_positive_pairs(values(true)),
                     _positive_pairs(values(pred)))


def _triangle(n):
    return n * (n - 1) // 2


def pairwise_negative(true, pred):
    """Return p_num, p_den, r_num, r_den over noncoreferent item pairs

    As used in calcualting BLANC (see Luo, Pradhan, Recasens and Hovy (2014).

    >>> pairwise_negative({1: {'a', 'b', 'c'}, 2: {'d'}},
    ...                   {1: {'b', 'c'}, 2: {'d', 'e'}})
    (2, 4, 2, 3)
    """
    true_pairs = _positive_pairs(values(true))
    pred_pairs = _positive_pairs(values(pred))
    n_pos_agreements = len(true_pairs & pred_pairs)

    true_mapping = sets_to_mapping(true)
    pred_mapping = sets_to_mapping(pred)
    extra_mentions = keys(true_mapping) ^ keys(pred_mapping)
    disagreements = {p for p in true_pairs ^ pred_pairs
                     if p[0] not in extra_mentions
                     and p[1] not in extra_mentions}

    n_common_mentions = len(keys(true_mapping) & keys(pred_mapping))
    n_neg_agreements = (_triangle(n_common_mentions) - n_pos_agreements -
                        len(disagreements))

    # Total number of negatives in each of pred and true:
    p_den = _triangle(len(pred_mapping)) - len(pred_pairs)
    r_den = _triangle(len(true_mapping)) - len(true_pairs)

    return n_neg_agreements, p_den, n_neg_agreements, r_den


def pairwise_slow(true, pred):
    p_den = sum(_triangle(len(pred_cluster)) for pred_cluster in values(pred))
    r_den = sum(_triangle(len(true_cluster)) for true_cluster in values(true))
    numerator = sum(_triangle(len(true_cluster & pred_cluster))
                    for true_cluster in values(true)
                    for pred_cluster in values(pred))
    return numerator, p_den, numerator, r_den


def pairwise_negative_slow(true, pred):
    trues = [len(true_cluster) for true_cluster in values(true)]
    preds = [len(pred_cluster) for pred_cluster in values(pred)]
    intersections = [[len(true_cluster & pred_cluster)
                      for true_cluster in values(true)]
                     for pred_cluster in values(pred)]
    n_pred = sum(preds)
    n_true = sum(trues)
    p_den = sum(a * (n_pred - a) for a in preds) // 2
    r_den = sum(a * (n_true - a) for a in trues) // 2
    row_sums = [sum(row) for row in intersections]
    N = sum(row_sums)
    col_sums = [sum(col) for col in zip(*intersections)]
    assert N == sum(col_sums)
    num = sum(n * (N - row_sums[row_idx] - col_sums[col_idx] + n)
              for row_idx, row in enumerate(intersections)
              for col_idx, n in enumerate(row)) // 2
    return num, p_den, num, r_den


def _slow_pairwise_negative(true, pred):
    """For testing comparison"""
    return _pairwise(_negative_pairs(values(true)),
                     _negative_pairs(values(pred)))


def _vilain(A, B_mapping):
    numerator = 0
    denominator = 0
    for cluster in A.values():
        corresponding = set()
        n_unaligned = 0
        for m in cluster:
            if m not in B_mapping:
                n_unaligned += 1
            else:
                corresponding.add(B_mapping[m])
        numerator += len(cluster) - n_unaligned - len(corresponding)
        denominator += len(cluster) - 1
    return numerator, denominator


def muc(true, pred):
    """The MUC evaluation metric defined in Vilain et al. (1995)

    This calculates recall error for each true cluster C as the number of
    response clusters that would need to be merged in order to produce a
    superset of C.
    """
    p_num, p_den = _vilain(pred, sets_to_mapping(true))
    r_num, r_den = _vilain(true, sets_to_mapping(pred))
    return p_num, p_den, r_num, r_den


####
def merge_eval_figures(evals, mode='add'):
    # input: a list of evaluation figures 
    # (p_num, p_den, r_num, r_den) or (precision, recall, f1)
    assert mode in ['add', 'average'], 'mode argument of merge_eval_figures not understood: %s' % mode
    if mode == 'add':
        return [sum(x) for x in zip(*evals)]

    elif mode == 'average':
        return [sum(x)/len(x) for x in zip(*evals)]
    
class DocumentSetEval():
    METRICS = {
        'muc': muc,
        'bcubed': b_cubed,
        'ceafe': entity_ceaf,
        'ceafm': mention_ceaf,
        'pairs': pairwise,
        'negpairs': pairwise_negative,
    }

    def __init__(self):
        self._reset()

    def _reset(self):
        self.total_eval_figures = {metric: [0, 0, 0, 0] for metric in self.METRICS.keys()}

    def eval_document(self, gold_clusters, predict_clusters):
        # calculate evaluation figures and update the total evaluation figures.
        eval_figures = self._eval_document(gold_clusters, predict_clusters)
        self.update_total_eval_figures(eval_figures)

        # print evaluation result of the current document.
        eval_scores = {metric: _prf(*figures) for metric, figures in eval_figures.items()}
        self.print_eval(eval_scores)

    def _eval_document(self, gold, sys):
        gold = {cid: set(mentions) for cid, mentions in gold.items()}
        sys = {cid: set(mentions) for cid, mentions in sys.items()}

        return {name: fn(gold, sys) for name, fn in self.METRICS.items()}

    def update_total_eval_figures(self, doc_eval):
        for metric, metric_figures in doc_eval.items():
            old_metric_figures = self.total_eval_figures[metric]
            self.total_eval_figures[metric] = merge_eval_figures([metric_figures, old_metric_figures], mode='add')

    def eval_document_set(self):
        # muc, bcubed, ceafe, ceafm
        eval_scores = { metric: _prf(*figures) 
                        for metric, figures in self.total_eval_figures.items() 
                        if metric in ['muc', 'bcubed', 'ceafe', 'ceafm'] }

        # blanc: average of pairs & negpairs scores
        eval_pairs = _prf(*self.total_eval_figures['pairs'])
        eval_neg_pairs = _prf(*self.total_eval_figures['negpairs'])
        eval_scores['blanc'] = merge_eval_figures([eval_pairs, eval_neg_pairs], mode='average')

        # calculate_average
        eval_scores['conll'] = merge_eval_figures([scores for metric, scores in eval_scores.items() 
                                                    if metric in ['muc', 'bcubed', 'ceafe', 'blanc']], mode='average')

        self.print_eval(eval_scores)

    def print_eval(self, eval_scores):
        print('\t'.join(['Metrics', 'P', 'R', 'F1']))
        for name, scores in eval_scores.items():
            p, r, f1 = scores
            print("%s\t%.2f\t%.2f\t%.2f" % (name, p*100, r*100, f1*100))

