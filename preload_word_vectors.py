import argparse
import pickle
import jsonlines
from Document import Document
from EmbedMap import EmbedMap
from utils import check_duplicate_file

def preload_word_vecs(input_files, output_file, word_vectors_file):
    # load complete word vectors mapping.
    embed_map = EmbedMap(word_vectors_file)

    # get list of lemmas that appeared in the input files.
    lemmas = set([])
    for input_file in input_files:
        if input_file == None:
            continue

        data_reader = jsonlines.open(input_file)
        for doc_data in data_reader.iter():
            doc = Document(doc_data)
            lemmas.update(m.lemma for m in doc.candidates.values())
            lemmas.update(m.lemma for m in doc.gold_mentions.values())
            
            for m in doc.candidates.values():
                lemmas.update(m.context)
            for m in doc.gold_mentions.values():
                lemmas.update(m.context)

    print("number of unique lemma: %s" % len(lemmas))

    # save the vectors of these lemmas to file.
    embed_map.save_word_vecs(output_file, lemmas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', action='store', dest='train_file')
    parser.add_argument('--dev_file', action='store', dest='dev_file')
    parser.add_argument('--test_file', action='store', dest='test_file')
    parser.add_argument('-o', '--output_file', action='store', dest='output_file')
    parser.add_argument('-w', '--word_vectors_file', action='store', dest='word_vectors_file')
    options = parser.parse_args()

    is_duplicate = check_duplicate_file(options.output_file)

    if not is_duplicate:
        input_files = [options.train_file, options.dev_file, options.test_file]
        preload_word_vecs(input_files, options.output_file, options.word_vectors_file)
