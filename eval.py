import argparse
import jsonlines
from Document import Document
from EmbedMap import EmbedMap
from models.WithinDocumentClassifier import load_model
from utils import *
from coref_metrics import DocumentSetEval


def cluster_mentions(mention_ids, predicts):
    mention_assigns, clusters = {}, []
    new_cluster_id = 0

    for j, this_mention_id in enumerate(mention_ids):
        candidate_pairs = ["%s-%s" % (mention_ids[i], this_mention_id) for i in range(j)]
        candidate_pairs = [pair for pair in candidate_pairs if pair in predicts]
        coref_scores = [predicts[pair_id] for pair_id in candidate_pairs]

        thres = 0.5
        if all(x < thres for x in coref_scores):
            mention_assigns[this_mention_id] = new_cluster_id
            clusters.append([this_mention_id])
            new_cluster_id += 1

        else:
            precedent_position = coref_scores.index(max(coref_scores))
            precedent_mention_id = mention_ids[precedent_position]
            coref_cluster_id = mention_assigns[precedent_mention_id]

            mention_assigns[this_mention_id] = coref_cluster_id
            clusters[coref_cluster_id].append(this_mention_id)

    # format into dictionary format for further processing
    clusters = {cid: mentions for cid, mentions in enumerate(clusters)}

    return clusters

def filter_cluster(clusters):
    """
    - Filter and reassign each cluster to multiple clusters with same subtype.
    - Also handle mentions with multiple subtypes.
    """
    filtered_clusters = []
    for cid, cluster in clusters.items():
        new_clusters = {}
        for mention in cluster:
            start_token, end_token, subtypes = mention.split('-')

            for subtype in subtypes.split('#'):
                new_mention = "%s-%s-%s" % (start_token, end_token, subtype)
                if subtype not in new_clusters:
                    new_clusters[subtype] = [new_mention]
                else:
                    new_clusters[subtype].append(new_mention)

        filtered_clusters += new_clusters.values()

    return {cid: cluster for cid, cluster in enumerate(filtered_clusters)}

def get_clusters(model_file, word_vectors_file, test_file, get_gold_clusters=False):
    # load model and vector mappings.
    model = load_model(model_file)
    embed_map = EmbedMap(word_vectors_file)

    # calculate system clusters
    gold_clusters, sys_clusters = {}, {}
    test_file_reader = jsonlines.open(test_file)
    for doc_data in test_file_reader.iter():
        doc = Document(doc_data, embed_map, remove_multi_word_mention=False)
        mention_pairs, ref_index = doc.generate_candidate_mention_pairs()
        mention_pairs_array, _ = convert_train_data(mention_pairs)

        # get predictions.
        predicts = [score[0] for score in model.predict(mention_pairs_array)]
        predicts = dict(zip(ref_index, predicts))

        # generate clusters. 
        doc_clusters = cluster_mentions(list(doc.candidates.keys()), predicts)

        gold_clusters[doc.doc_id] = doc.gold_clusters
        sys_clusters[doc.doc_id] = doc_clusters

    if get_gold_clusters:
        return gold_clusters, sys_clusters
    else:
        return sys_clusters

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_file', action='store', dest='test_file')
    parser.add_argument('-w', '--word_vectors_file', action='store', dest='word_vectors_file')
    parser.add_argument('-m', '--model_file', action='store', dest='model_file')
    options = parser.parse_args()

    test_eval = DocumentSetEval()
    gold_clusters, sys_clusters = get_clusters(options.model_file, options.word_vectors_file, options.test_file, get_gold_clusters=True)

    for doc_id in gold_clusters.keys():
        gold, sys = gold_clusters[doc_id], sys_clusters[doc_id]

        # print evaluation results of the current file.
        print('#####', doc_id)
        test_eval.eval_document(filter_cluster(gold), filter_cluster(sys))

    # print evaluation results.
    print('##### all documents')
    test_eval.eval_document_set()

