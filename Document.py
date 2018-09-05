import sys
import numpy as np
import itertools
from collections import OrderedDict
from utils import *
from Mention import Mention

class Document():
    def __init__(self, raw_data, embed_map=None, remove_multi_word_mention=True, debug=False):
        # record document ID and other initialization options.
        self.doc_id = get_doc_id(raw_data['doc_key'])
        self.debug = debug
        self.embed_map = embed_map

        # process raw sentence.
        self._process_sentences(raw_data['sentences'], raw_data['sentences_lemma'], raw_data['sentences_pos'])

        # process mention candidates.
        if 'subtypes' not in raw_data:  # train file.
            self.candidates = {}

        else:
            self._process_candidates(raw_data['candidates'], raw_data['subtypes'])
            self._set_mentions(self.candidates, raw_data['candidate_arguments'])

        # process gold mention clusters.
        self._process_gold_clusters(raw_data['gold_event_mentions'], raw_data['gold_clusters'], remove_multi_word_mention)
        self._set_mentions(self.gold_mentions, raw_data['candidate_arguments'])

        #
        self.mentions = list(set(self.candidates.keys()) | set(self.gold_mentions.keys()))
        self.set_anaphoricity()

    def _process_sentences(self, sentences, lemmas, pos):
        """
        - Construct self.tokens, self.lemmas, self.pos
        - Construct self.chars:
            - [start_char, end_char] of each token
        """
        self.sentences, escape_tokens = build_sentence_structure(sentences)

        self.tokens = flatten_to_tokens(sentences)
        self.chars = tokens_to_chars(self.tokens, escape_tokens)
        self.lemmas = flatten_to_tokens(lemmas)
        self.pos = flatten_to_tokens(pos)

        assert len(self.tokens) == len(self.chars)
        assert len(self.tokens) == len(self.pos)
        assert len(self.tokens) == len(self.lemmas)

    def _process_gold_clusters(self, gold_mentions, gold_clusters, remove_multi_word_mention=True):
        """
        - Construct self.gold_mentions
            - { mention-id: mention-object }
            - mention-id: #START_TOKEN-#END_TOKEN-#SUBTYPE
            - mentions that appears in at least one of the gold clusters
        - Construct self.gold_clusters:
            { cluster-id: list of mention ids in this cluster }
        """
        # parse gold mentions.
        self.gold_mentions = OrderedDict()
        for start_token, end_token, subtype in gold_mentions:
            if remove_multi_word_mention and start_token != end_token:
                continue

            mention_id = '%s-%s-%s' % (start_token, end_token, subtype)
            sent_id = get_sentence_num(self.sentences, start_token)
            self.gold_mentions[mention_id] = Mention(self.doc_id, mention_id, sent_id)

        # parse gold clusters.
        self.gold_clusters = {}
        for cluster_id, cluster in enumerate(gold_clusters):
            cluster_mention_ids = ['%s-%s-%s' % tuple(mention) for mention in cluster] 
            cluster_mention_ids = [ii for ii in cluster_mention_ids if ii in self.gold_mentions]

            # update cluster_id attribute of cluster mentions.
            for mention_id in cluster_mention_ids:
                mention = self.gold_mentions[mention_id]
                mention.set_cluster_id(cluster_id)

                # also update candidate mentions.
                if mention_id in self.candidates:
                    mention = self.candidates[mention_id]
                    mention.set_cluster_id(cluster_id)

            # update self.gold_clusters
            if cluster_mention_ids != []:
                self.gold_clusters[cluster_id] = cluster_mention_ids

    def _process_candidates(self, candidates, candidate_subtypes):
        """
        - set self.candidates 
            - { mention-id : mention-obj }
            - mention-id: #START_TOKEN-#END_TOKEN-#SUBTYPE
            - mentions idenitifed by the upstream trigger detection system.
        """
        self.candidates = OrderedDict()
        for candi_idx, candi_subtype in zip(candidates, candidate_subtypes):
            # remove non-trigger words.
            if candi_subtype == 'null':
                continue

            # remove multi-word candidate mentions.
            if candi_idx[0] != candi_idx[1]:
                continue

            mention_id = '%s-%s-%s' % (candi_idx[0], candi_idx[1], candi_subtype)
            sent_id = get_sentence_num(self.sentences, candi_idx[0])
            self.candidates[mention_id] = Mention(self.doc_id, mention_id, sent_id)

    def _set_mentions(self, mention_dict, arguments, window_size=3):
        # set lemma, pos, context, antecedent.
        for index, (mention_id, mention) in enumerate(mention_dict.items()):
            mention_pos = mention.start_token
            lemma = self.lemmas[mention_pos]
            PoS = self.pos[mention_pos]
            context = self.get_context_of_idx(mention_pos, window_size)

            mention.set_mention_feature(lemma, PoS, context)
            mention.antecedent = {mid: mention.lemma for mid, mention in list(mention_dict.items())[:index]}

        # set arguments of candidate mentions.
        for arg_info in arguments:
            arg = self.tokens[arg_info[2]: arg_info[3] + 1]
            arg_head = self.tokens[arg_info[4]: arg_info[5] + 1]
            arg_type, arg_coref_id = arg_info[-2:]
            arg_char_idx = [self.chars[arg_info[2]][0], self.chars[arg_info[3]][1]]

            for mention_id in find_mention_ids(mention_dict.keys(), arg_info[0], arg_info[1]):
                mention = mention_dict[mention_id]
                mention.set_arguments(arg, arg_head, arg_type, arg_coref_id, arg_char_idx)

    def set_anaphoricity(self):
        gold_non_anaphor = get_nanaphoric_mentions(self.gold_mentions)
        for mid, mention in self.gold_mentions.items():
            mention.anaphoricity = False if mid in gold_non_anaphor else True

        # candidate
        candi_non_anaphor = get_nanaphoric_mentions(self.candidates)
        for mid, mention in self.candidates.items():
            mention.anaphoricity = False if mid in candi_non_anaphor else True

    def get_context_of_idx(self, target_idx, window_size=3):
        # calculate the indexes of the context array and padding size.
        L = len(self.tokens)
        context_idx_0, front_pad = (target_idx - window_size, 0) if target_idx >= window_size \
                                        else (0, window_size - target_idx)
        context_idx_1, post_pad = (target_idx + window_size + 1, 0) if target_idx < L - window_size \
                                        else (L, window_size - L + target_idx + 1)

        # construct the context array, with NULL padding.
        context = ['NULL' for _ in range(front_pad)] + \
                  self.tokens[context_idx_0: context_idx_1] + \
                  ['NULL' for _ in range(post_pad)]

        # check size
        assert len(context) == 2 * window_size + 1, 'context vector has wrong shape, target index %s / %s' % (target_idx, L)
        return context
    
    def get_visualize_data(self):
        doc_data = {}
        doc_data['entities'] = []
        doc_data['triggers'] = []
        doc_data['events'] = []

        # text
        doc_data['text'] = self.get_document_content()

        # entities: arguments, triggers: cadidate-mention, events: define argument links 
        for m_id in self.mentions:
            if m_id in self.gold_mentions:
                mention, ev_type = self.gold_mentions[m_id], 'Gold'
            else:
                mention, ev_type = self.candidates[m_id], 'Candidate'

            # set triggers
            mention_name = 'M%s' % m_id
            char_idx = [self.chars[mention.start_token][0], self.chars[mention.end_token][1]]
            doc_data['triggers'].append([mention_name, ev_type, [char_idx]])

            # set entities (arguments)
            arcs = []
            for arg_type, arg in mention.args.items():
                arg_name = 'A%s-%s' % tuple(arg.token_idx)
                arcs.append([arg_type, arg_name])

                arg_rep = [arg_name, 'Argument', [arg.token_idx]]
                if arg_rep not in doc_data['entities']:
                    doc_data['entities'].append(arg_rep)

            # set events (argument arcs)
            event_name = 'E%s' % m_id
            events_rep = [event_name, mention_name, arcs]
            doc_data['events'].append(events_rep)

        return doc_data

    def get_document_content(self):
        content = ''
        for _, (start_idx, fin_idx) in self.sentences.items():
            content += ' '.join(self.tokens[start_idx:fin_idx])
            content += '\n'
        return content

    def get_cluster_data(self):
        cluster_data = []
        for cluster_ids in self.gold_clusters.values():
            cluster_ids = ["E%s" % ii for ii in cluster_ids]
            cluster_data.append(cluster_ids)

        return cluster_data

    def get_surface_data(self):
        surface_data = {}
        for m_id in self.mentions:
            event_name = 'E%s' % m_id
            mention = self.gold_mentions[m_id] if m_id in self.gold_mentions else self.candidates[m_id]
            surface_data[event_name] = mention.get_mention_detail()

        return surface_data

    def generate_gold_mention_pairs(self, with_ref_index=False):
        """
        genrate training samples from gold mentions.
        """
        assert self.embed_map != None, 'The embedding map is not set.'
        train_data = [m1.generate_joint_data(m2, self.embed_map)
                            for m1, m2 in itertools.combinations(self.gold_mentions.values(), 2)]

        ref_index = ["%s-%s" % (m1_id, m2_id)
                            for m1_id, m2_id in itertools.combinations(self.gold_mentions.keys(), 2)]

        assert len(ref_index) == len(train_data)

        if with_ref_index:
            return train_data, ref_index

        else:
            return train_data

    def generate_joint_training_pairs(self, with_ref_index=False):
        assert self.embed_map != None, 'The embedding map is not set.'
        train_data = [m1.generate_joint_data(m2, self.embed_map)
                            for m1, m2 in itertools.combinations(self.gold_mentions.values(), 2)]

        ref_index = ["%s-%s" % (m1_id, m2_id)
                            for m1_id, m2_id in itertools.combinations(self.gold_mentions.keys(), 2)]

        assert len(ref_index) == len(train_data)

        if with_ref_index:
            return train_data, ref_index

        else:
            return train_data


    def generate_candidate_mention_pairs(self, with_ref_index=True):
        """
        genrate test samples from candidate mentions.
        """
        assert self.embed_map != None, 'The embedding map is not set.'
        test_data = [m1.generate_joint_data(m2, self.embed_map)
                            for m1, m2 in itertools.combinations(self.candidates.values(), 2)]

        ref_index = ["%s-%s" % (m1_id, m2_id)
                            for m1_id, m2_id in itertools.combinations(self.candidates.keys(), 2)]

        assert len(ref_index) == len(test_data)

        if with_ref_index:
            return test_data, ref_index

        else:
            return test_data

    def generate_dev_mention_pairs(self):
        assert self.embed_map != None, 'The embedding map is not set.'
        train_data = [m1.generate_joint_data(m2, self.embed_map)
                            for m1, m2 in itertools.combinations(self.gold_mentions.values(), 2)]

        test_data = [m1.generate_joint_data(m2, self.embed_map)
                            for m1, m2 in itertools.combinations(self.candidates.values(), 2)
                            if m1.cluster_id == -1 or m2.cluster_id == -1]

        return train_data + test_data


    def generate_subtype_mention_pairs(self):
        """
        genrate test samples from candidate mentions, but only with mentions of the same type.
        """
        assert self.embed_map != None, 'The embedding map is not set.'
        test_data = [m1.generate_siamese(m2, self.embed_map)
                            for m1, m2 in itertools.combinations(self.candidates.values(), 2)
                            if m1.subtype == m2.subtype]

        ref_index = ["%s-%s" % (m1.id, m2.id)
                            for m1, m2 in itertools.combinations(self.candidates.values(), 2)
                            if m1.subtype == m2.subtype]

        assert len(ref_index) == len(test_data)
        return test_data, ref_index

    def generate_gold_anaphor_data(self, with_ref_index=False):
        assert self.embed_map != None, 'The embedding map is not set.'
        data = [mention.generate_anaphor_data(self.embed_map) 
                            for mention in self.gold_mentions.values()]

        ref_index = list(self.gold_mentions.keys())
        assert len(ref_index) == len(data)

        if with_ref_index:
            return data, ref_index
        else:
            return data

    def generate_candidate_anaphor_data(self, with_ref_index=False):
        # deal with the -1 cluster.
        data = [mention.generate_anaphor_data(self.embed_map)
                            for mention in self.candidates.values()]

        ref_index = list(self.candidates.keys())
        assert len(ref_index) == len(data)

        if with_ref_index:
            return data, ref_index
        else:
            return data

if __name__ == "__main__":
    import jsonlines
    data_file = "./data/formatted/english.E51.test.withsubtype.jsonlines"
    data_reader = jsonlines.open(data_file)
    
    for doc_data in data_reader.iter():
        doc = Document(doc_data)
        for mid, m in list(doc.gold_mentions.items()):
            print(mid, m.anaph_h)
        break
