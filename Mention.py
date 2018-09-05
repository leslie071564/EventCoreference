import os
import numpy as np
from collections import namedtuple
Argument = namedtuple('Argument', 'arg arg_head coref_id token_idx')

# define the mappings of argument types.
arg0s = ['agent', 'entity', 'giver', 'attacker']
arg1s = ['victim', 'person', 'artifact', 'target', 'audience', 'recipient']
locs = ['orgin', 'destination', 'place']
times = ['time']
arg_type_convert = [(raw_type, 'arg0') for raw_type in arg0s] + \
                   [(raw_type, 'arg1') for raw_type in arg1s] + \
                   [(raw_type, 'loc') for raw_type in locs] + \
                   [(raw_type, 'time') for raw_type in times]
arg_type_convert = dict(arg_type_convert)


class Mention():
    def __init__(self, doc_id, mention_id, sentence_id):
        self.doc_id = doc_id
        self.id = mention_id
        self.sentence_id = sentence_id
        self.start_token, self.end_token, self.subtype = mention_id.split('-')
        self.start_token, self.end_token = int(self.start_token), int(self.end_token)

        self.args = {}
        self.cluster_id = -1
        self.anaphoricity = None
        self.antecedent = {}

    def set_mention_feature(self, lemma, pos, context_words):
        self.lemma = lemma.lower()
        self.pos = pos
        self.context = context_words

    def set_arguments(self, arg, arg_head, raw_type, arg_coref_id, arg_token_idx):
        """
        take in raw argument and argument type, convert arg_type into the following category:
            - arg0
            - arg1
            - loc
            - time
        """
        # convert argument type.
        if raw_type not in arg_type_convert:
            return

        arg_type = arg_type_convert[raw_type]

        # set argument and argument head word.
        arg_tuple = Argument(arg, '_'.join(arg_head), arg_coref_id, arg_token_idx)
        self.args[arg_type] = arg_tuple

    def set_cluster_id(self, cluster_id):
        self.cluster_id = cluster_id

    def get_mention_detail(self):
        args_detail = []
        for arg_type in ['arg0', 'arg1', 'loc', 'time']:
            if arg_type not in self.args:
                args_detail.append('')
                continue

            this_arg = self.args[arg_type]
            if len(this_arg.arg) <= 3:
                args_detail.append('%s_%s' % ('_'.join(this_arg.arg), this_arg.coref_id))

            else:   # if argument is too long, use argument head instead.
                args_detail.append('@%s_%s' % (this_arg.arg_head, this_arg.coref_id))

        assert len(args_detail) == 4
        return [self.lemma, self.pos] + args_detail + [self.cluster_id]

    def get_mention_feature_array(self, embed_map):
        lemma_vec = embed_map.lookup_word_vec(self.lemma)
        pos_vec = embed_map.lookup_pos_vec(self.pos)

        return np.concatenate([lemma_vec, pos_vec])

    def get_context_feature_array(self, embed_map):
        context_vec = [embed_map.lookup_word_vec(w) for w in self.context]
        return np.array(context_vec)

    def get_mean_context_feature(self, embed_map):
        context_vec = [embed_map.lookup_word_vec(w) for w in self.context]
        context_vec = np.array([v for v in context_vec if not np.array_equal(v, embed_map.default_word_vec)])
        return np.mean(context_vec, axis=0)

    def compare_arg(self, mention_2):
        compare_result = []

        for arg_type in ['arg0', 'arg1', 'loc', 'time']:
            if arg_type not in self.args or arg_type not in mention_2.args:
                if_same_arg = False

            else:
                if_same_arg = (self.args[arg_type].coref_id == mention_2.args[arg_type].coref_id)

            compare_result.append(int(if_same_arg))
        return compare_result

    def detailed_compare_arg(self, mention_2):
        compare_result = []

        for arg_type in ['arg0', 'arg1', 'loc', 'time']:
            if arg_type not in self.args or arg_type not in mention_2.args:
                compare_result += [0, 1, 0]
                continue
            
            if self.args[arg_type].coref_id == mention_2.args[arg_type].coref_id:
                compare_result += [1, 0, 0]
                continue

            else:
                compare_result += [0, 0, 1]

        return compare_result

    def get_dist_feat(self, mention_2, max_dist=4):
        sentence_dist = abs(self.sentence_id - mention_2.sentence_id)
        sentence_dist = max_dist if sentence_dist > max_dist else sentence_dist

        dist_feat = [0 for _ in range(max_dist + 1)]
        dist_feat[sentence_dist] = 1
        return dist_feat

    def generate_train_data(self, mention_2, embed_map):
        # generate lemma_features
        EM_1 = self.get_mention_feature_array(embed_map)
        EM_2 = mention_2.get_mention_feature_array(embed_map)

        # generate overlapping-argument features
        ARG = np.array(self.compare_arg(mention_2) + self.get_dist_feat(mention_2)) 

        # decide if m1 & m2 is coreferent.
        Y = (self.cluster_id == mention_2.cluster_id and self.cluster_id != -1)

        return (EM_1, EM_2, ARG, Y)

    def generate_siamese(self, mention_2, embed_map):
        # two mention representations, one auxiliary feature vector.
        # generate lemma_features
        EM_1 = self.get_mention_feature_array(embed_map)
        EM_2 = mention_2.get_mention_feature_array(embed_map)

        CON_1 = self.get_mean_context_feature(embed_map)
        CON_2 = mention_2.get_mean_context_feature(embed_map)
        CON_diff = CON_1 - CON_2

        # generate overlapping-argument features
        #AUX = np.array(self.detailed_compare_arg(mention_2) + self.get_dist_feat(mention_2))
        AUX = np.array(self.compare_arg(mention_2) + self.get_dist_feat(mention_2))

        # decide if m1 & m2 is coreferent.
        Y = (self.cluster_id == mention_2.cluster_id and self.cluster_id != -1)

        return (EM_1, EM_2, CON_diff, AUX, Y)

    def generate_anaphor_data(self, embed_map):
        EM = self.get_mention_feature_array(embed_map)
        CON = self.get_context_feature_array(embed_map)
        AUX = int(self.lemma in self.antecedent.values())
        
        Y = int(self.anaphoricity)

        return (EM, CON, AUX, Y)

    def generate_joint_data(self, mention_2, embed_map):
        # generate lemma_features
        EM_1 = self.get_mention_feature_array(embed_map)
        EM_2 = mention_2.get_mention_feature_array(embed_map)

        # auxiliary feature of coref.
        AUX_C = np.array(self.compare_arg(mention_2) + self.get_dist_feat(mention_2))

        CON_1 = self.get_mean_context_feature(embed_map)
        CON_2 = mention_2.get_mean_context_feature(embed_map)
        CON_diff = CON_1 - CON_2

        # auxiliary feature of anaph
        CON = self.get_context_feature_array(embed_map)
        AUX_A = int(self.lemma in self.antecedent.values())

        # decide if m1 & m2 is coreferent.
        Y_C = (self.cluster_id == mention_2.cluster_id and self.cluster_id != -1)
        Y_A = int(self.anaphoricity)

        return (EM_1, EM_2, CON_diff, CON, AUX_C, AUX_A, Y_C, Y_A)
