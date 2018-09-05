import jsonlines
from Document import Document
from coref_metrics import DocumentSetEval

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

def get_best_clusters(doc):
    best_cluster_possible = {}
    for mention_id, mention in doc.candidates.items():
        cid = mention.cluster_id
        if cid not in best_cluster_possible:
            best_cluster_possible[cid] = [mention_id]
        else:
            best_cluster_possible[cid].append(mention_id)

    best_cluster_possible = [cluster for cid, cluster in best_cluster_possible.items() if cid != -1]\
                                + [[mid] for mid in best_cluster_possible.get(-1, [])]
    best_cluster_possible = {cid: cluster for cid, cluster in enumerate(best_cluster_possible)}
    return best_cluster_possible

def get_majority_cluster(doc, without_conflict=False):
    majority_cluster = {}
    for mention_id, mention in doc.candidates.items():
        lemma_key = mention.lemma
        if lemma_key not in majority_cluster:
            majority_cluster[lemma_key] = [mention_id]

        else:
            majority_cluster[lemma_key].append(mention_id)

    majority_cluster = {cid: cluster for cid, cluster in enumerate(majority_cluster.values())}
    return majority_cluster


if __name__ == "__main__":
    #test_doc = "./data/formatted/english.E51.test.withsubtype.jsonlines"
    test_doc = "./data/formatted/english.E64.dev.withsubtype.jsonlines"
    test_reader = jsonlines.open(test_doc)
    test_eval = DocumentSetEval()
    
    for doc_data in test_reader.iter():
        doc = Document(doc_data)
        #sys_cluster = get_best_clusters(doc)
        sys_cluster = get_majority_cluster(doc)

        # print evaluation results of the current file.
        print('#####', doc.doc_id)
        test_eval.eval_document(filter_cluster(doc.gold_clusters), filter_cluster(sys_cluster))

    # print evaluation results.
    print('##### all documents')
    test_eval.eval_document_set()

