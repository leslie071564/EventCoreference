import os
import sys
import pickle
import numpy as np

tag_set = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', \
           'JJ', 'JJR', 'JJS', 'LS', 'MD', \
           'NN', 'NNS', 'NNP', 'NNPS', \
           'PDT', 'POS', 'PRP', 'PRP$',\
           'RB', 'RBR', 'RBS', 'RP', \
           'SYM', 'TO', 'UH', \
           'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', \
           'WDT', 'WP', 'WP$', 'WRB']

class EmbedMap():
    def __init__(self, word_vectors_file):
        self._set_word_vecs(word_vectors_file)
        self._set_pos_vecs()

    def _set_word_vecs(self, word_vectors_file):
        # check file existence.
        assert os.path.isfile(word_vectors_file), 'file not existed: %s' % word_vectors_file

        # check file type.
        _, file_type = os.path.splitext(word_vectors_file)
        assert file_type in ['.pkl', '.bin'], 'file format not supported: %s' % file_type

        # extract word vectors from file.
        sys.stderr.write("loading word vectors from: %s\n" % word_vectors_file)
        if file_type == '.pkl':
            w2v_mapping = pickle.load(open(word_vectors_file, 'rb'))

        elif file_type == '.bin':
            from gensim.models.keyedvectors import KeyedVectors
            w2v_mapping = KeyedVectors.load_word2vec_format(word_vectors_file, binary=True)

        self.word_vecs = w2v_mapping
        self.default_word_vec = np.zeros((300,))

    def _set_pos_vecs(self):
        pos_array = np.eye(len(tag_set), dtype=float)
        self.pos_vecs = {tag: pos_array[tag_id]
                           for tag_id, tag in enumerate(tag_set)}
        self.default_pos_vec = np.zeros((len(tag_set),))

    def lookup_word_vec(self, word, debug=False):
        try:
            word_vec = self.word_vecs[word]

        except:
            word_vec = self.default_word_vec
            if debug:
                sys.stderr.write('word %s not found.\n' % word)

        return word_vec

    def lookup_pos_vec(self, pos, debug=False):
        if debug and pos not in self.pos_vecs:
            sys.stderr.write('POS %s not found.\n' % pos)

        return self.pos_vecs.get(pos, self.default_pos_vec)

    def save_word_vecs(self, output_file, word_list):
        word_vecs = {word: self.lookup_word_vec(word) for word in word_list}

        with open(output_file, 'wb') as f:
            pickle.dump(word_vecs, f, protocol=pickle.HIGHEST_PROTOCOL)
            sys.stderr.write('write the word vectors to file: %s\n' % output_file)

