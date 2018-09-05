import os
import sys
import argparse
import jsonlines
from Document import Document
from utils import save_to_json, get_json_data, check_duplicate_dir

def update_corpus_ids(doc_id):
    corpus_ids_file = "./visualization/corpus/corpus_ids.json"
    corpus_ids = get_json_data(corpus_ids_file) if os.path.isfile(corpus_ids_file) else []

    if doc_id not in corpus_ids:
        corpus_ids = [doc_id] + corpus_ids
        save_to_json(corpus_ids_file, corpus_ids)
        print('corpus ids updated: %s' % doc_id)

def genrate_gold_gui_data(corpus_dir, doc_id, data_file):
    data_reader = jsonlines.open(data_file)

    # handle the case that the doc_id already exists.
    if check_duplicate_dir(corpus_dir):
        sys.exit()

    doc_ids = []
    for doc_dict in data_reader.iter():
        doc = Document(doc_dict)
        doc_ids.append(doc.doc_id)

        # doc data
        doc_data = doc.get_visualize_data()
        doc_data_file = "%s/span/%s.json" % (corpus_dir, doc.doc_id)
        save_to_json(doc_data_file, doc_data)

        # surface data
        surface_data = doc.get_surface_data()
        surface_data_file = "%s/detail/%s.json" % (corpus_dir, doc.doc_id)
        save_to_json(surface_data_file, surface_data)

        # cluster data
        cluster_data = doc.get_cluster_data()
        cluster_data_file = "%s/coref/%s.json" % (corpus_dir, doc.doc_id)
        save_to_json(cluster_data_file, cluster_data)

    # doc ids and corpus ids.
    doc_ids_file = "%s/doc_ids.json" % corpus_dir
    save_to_json(doc_ids_file, doc_ids)
    update_corpus_ids(doc_id)

def generate_test_gui_data(corpus_dir, exp_id, test_file, model_file, word_vectors_file):
    from eval import get_clusters
    sys_clusters = get_clusters(model_file, word_vectors_file, test_file, get_gold_clusters=False)

    for doc_id, doc_clusters in sys_clusters.items():
        # save cluster to file.
        output_file = "%s/coref/%s.json" % (corpus_dir, doc_id)
        save_to_json(output_file, get_cluster_data_for_gui(doc_clusters))

    # doc ids and corpus ids.
    doc_ids_file = "%s/doc_ids.json" % corpus_dir
    save_to_json(doc_ids_file, list(sys_clusters.keys()))

    update_corpus_ids(exp_id)

def get_cluster_data_for_gui(clusters):
    cluster_data = [['E%s' % mention_id for mention_id in mentions] for mentions in clusters.values()]
    return cluster_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', action='store_true', dest='gold')
    parser.add_argument('-d', '--data_file', action='store', dest='data_file')
    parser.add_argument('-e', '--doc_id', action='store', dest='doc_id')
    parser.add_argument('-o', '--corpus_dir', action='store', dest='corpus_dir')

    parser.add_argument('--test', action='store_true', dest='test')
    parser.add_argument('-w', '--word_vectors_file', action='store', dest='word_vectors_file')
    parser.add_argument('-m', '--model_file', action='store', dest='model_file')
    options = parser.parse_args()

    if options.gold:
        genrate_gold_gui_data(options.corpus_dir, options.doc_id, options.data_file)

    if options.test:
        generate_test_gui_data(options.corpus_dir, options.doc_id, options.data_file, options.model_file, options.word_vectors_file)
