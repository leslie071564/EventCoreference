import os
import numpy as np
import json

def remove_duplicate(input_list):
    seen = []
    seen_add = seen.append
    return [x for x in input_list if not (x in seen or seen_add(x))]

def flatten_to_tokens(sentences):
    tokens = [token for sent in sentences for token in sent]
    return tokens

def tokens_to_chars(tokens, escape_tokens):
    start_char_idx, fin_char_idx = [0], []

    last_idx = 0
    for i, w in enumerate(tokens):
        wl = len(w)
        if i in escape_tokens:
            start_char_idx.append(-1)
            fin_char_idx.append(-1)
        else:
            start_char_idx.append(last_idx + wl + 1)
            fin_char_idx.append(last_idx + wl)
            last_idx = last_idx + wl + 1

    return list(zip(start_char_idx, fin_char_idx))

def sentences_to_tokens(sentences):
    start_token_idx, fin_token_idx = [0], []
    for sent in sentences:
        next_boundary = start_token_idx[-1] + len(sent)
        start_token_idx.append(next_boundary)
        fin_token_idx.append(next_boundary)

    return list(zip(start_token_idx, fin_token_idx))

def get_sentence_num(sentence_token_mapping, target_token):
    for sent_num, (start_id, fin_id) in sentence_token_mapping.items():
        if start_id <= target_token and target_token < fin_id:
            return sent_num

def convert_to_ndarray(list_of_array):
    return np.stack(list_of_array, axis=0)

def save_to_json(json_file, data):
    with open(json_file, 'w') as handle:
        json.dump(data, handle)

def get_json_data(json_file):
    with open(json_file, 'r') as handle:
        return json.load(handle)

def check_duplicate_dir(some_dir):
    # check if a directory already exists, and prompt user for overwrite options.
    if not os.path.isdir(some_dir) or not os.listdir(some_dir):
        return False

    # if directory exists, promt user for overwrite options.
    if_overwrite = input('Overwrite contents in %s?[y/n]: ' % some_dir)
    if if_overwrite == 'y':
        return False

    elif if_overwrite == 'n':
        return True

    else:
        sys.stderr.write('user-input not understood, assume not overwriting.\n')
        return True

def check_duplicate_file(some_file):
    if not os.path.isfile(some_file):
        return False

    # if directory exists, promt user for overwrite options.
    if_overwrite = input('Overwrite contents in %s?[y/n]: ' % some_file)
    if if_overwrite == 'y':
        return False

    elif if_overwrite == 'n':
        return True

    else:
        sys.stderr.write('user-input not understood, assume not overwriting.\n')
        return True

def find_mention_ids(mention_ids, start_token, end_token):
    return_mention_ids = []
    prefix = "%s-%s" % (start_token, end_token)
    for iid in mention_ids:
        if iid.startswith(prefix):
            return_mention_ids.append(iid)

    #if len(return_mention_ids) > 1:
    #    print('multiple events in same loc: (%s, %s)' % (start_token, end_token))
    return return_mention_ids

def get_doc_id(doc_path):
    return os.path.basename(doc_path)

def get_parent_dir(dir_path):
    return os.path.abspath(os.path.join(dir_path, '..'))

def convert_train_data(raw_train_data, y_len=1):
    if raw_train_data == []:
        return ([], [])

    data_channels = [convert_to_ndarray(d) for d in zip(*raw_train_data)]

    Xs = data_channels[:-y_len]
    Y = data_channels[-y_len:]
    if y_len == 1:
        Y = Y[0]

    return (Xs, Y)

def is_symbolic_line(line):
    if type(line) == list:
        return not any(c.isalpha() for c in ' '.join(line))
        
    return not any(c.isalpha() for c in line)

def build_sentence_structure(sentences):
    # remove sentence that consists of symbols.
    filtered_sentences = []
    topic_line = None
    for s in sentences:
        if is_symbolic_line(s):
            filtered_sentences.append([])
            continue

        filtered_sentences.append([s])
        if topic_line == None:
            topic_line = ' '.join(s)

    # seperate repetitive topic lines.
    for i, s in enumerate(filtered_sentences):
        if len(s) == 0:
            continue

        ss = ' '.join(s[0])
        seperate_sentences = []
        while ss.startswith(topic_line):
            seperate_sentences.append(topic_line.split())
            ss = ss[len(topic_line) + 1:]
        if len(ss) != 0:
            seperate_sentences.append(ss.split())
        filtered_sentences[i] = seperate_sentences

    # maintain the sentence indexes
    raw_indexes = {ii: sent_idx for ii, sent_idx in enumerate(sentences_to_tokens(sentences))}
    s_indexes = []
    escape_tokens = []
    for i, s in enumerate(filtered_sentences):
        if len(s) == 0:
            escape_tokens += list(range(*raw_indexes[i]))
            continue

        if len(s) == 1:
            s_indexes.append(raw_indexes[i])
            continue

        offset, _ = raw_indexes[i]
        for sub in s:
            s_indexes.append([offset, offset + len(sub)])
            offset += len(sub)
    return {ii: sent_idx for ii, sent_idx in enumerate(s_indexes)}, escape_tokens

def get_nanaphoric_mentions(gold_mentions):
    first_mentions = {} # cluster-id: (mention_name, start-token)
    for mid, mention in gold_mentions.items():
        cluster_id = mention.cluster_id

        if cluster_id == -1:
            continue

        if cluster_id not in first_mentions:
            first_mentions[cluster_id] = (mid, mention.start_token)

        elif mention.start_token < first_mentions[cluster_id][1]:
            first_mentions[cluster_id] = (mid, mention.start_token)

    return [mid for (mid, start_token) in first_mentions.values()]
