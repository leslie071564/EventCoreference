import os
import jsonlines
import argparse
import numpy as np
import keras
from random import shuffle
from Document import Document
from EmbedMap import EmbedMap
from models.AnaphoricityClassifier import get_model, load_model
from utils import convert_train_data

def generate_raw_data(input_file, embed_map, gen_type='gold'):
    print('loading data from %s' % input_file)
    data_reader = jsonlines.open(input_file)

    raw_data = []
    for doc_data in data_reader.iter():
        doc = Document(doc_data, embed_map)
        if gen_type == 'gold':
            raw_data += doc.generate_gold_anaphor_data()
        else:
            raw_data += doc.generate_candidate_anaphor_data()

    print("---> total number of training pairs: %s" % len(raw_data))
    return raw_data

def generate_train_data(input_file, word_vectors_file, N_dev=200):
    # load word vectors and PoS one-hot vectors.
    embed_map = EmbedMap(word_vectors_file)
    train_data = generate_raw_data(input_file, embed_map)

    # shuffle the training data and convert to seperate numpy arrays.
    shuffle(train_data)

    data_X, data_y = convert_train_data(train_data)
    train_X, train_y = [cn[:-N_dev] for cn in data_X], data_y[:-N_dev]
    dev_X, dev_y = [cn[-N_dev:] for cn in data_X], data_y[-N_dev:]

    return {'train_X': train_X, 'train_y': train_y,
            'dev_X': dev_X, 'dev_y': dev_y}

def generate_test_data(input_file, word_vectors_file):
    # load word vectors and PoS one-hot vectors.
    embed_map = EmbedMap(word_vectors_file)
    test_data = generate_raw_data(input_file, embed_map, gen_type='candi')

    data_X, data_y = convert_train_data(test_data)
    return data_X, data_y

def train_model(train_data, dev_data, model_file, lr=0.001, batch_size=128, epochs=10):
    # load train and dev data.
    train_X, train_y = train_data
    dev_X, dev_y = dev_data

    # load model
    print('load model structure')
    model = get_model(silent=True)

    # compile model
    print('---> complile model (learn-rate: %s)' % lr)
    opt = keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # train model and save the best model to file.
    print('---> train model (#mb: %s, #epoch: %s)' % (batch_size, epochs))
    checkpoint = keras.callbacks.ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
    hist = model.fit(train_X, train_y, validation_data=(dev_X, dev_y),
              batch_size=batch_size, epochs=epochs, shuffle=True,
              verbose=1, callbacks=[checkpoint, earlystop])

def eval(models, x, gold_Y, return_hidden_vals=False):
    # load model
    model, hidden_model = models

    # make predictions
    predict_Y = np.round(model.predict(x)).reshape(gold_Y.shape)

    if return_hidden_vals:
        hidden_val = hidden_model.predict(x)
        return hidden_val

    else:
        # calculate precision, recall, F1 scores.
        correct_predict = np.count_nonzero(predict_Y * gold_Y)
        gold_anaphor_num = np.count_nonzero(gold_Y)
        predict_anaphor_num = np.count_nonzero(predict_Y)

        prec = correct_predict / predict_anaphor_num
        recall = correct_predict / gold_anaphor_num
        f1 = 2 * prec * recall / (prec + recall)

        # print evaluation scores.
        print('precsion: %.4f' % prec)
        print('recall  : %.4f' % recall)
        print('F1      : %.4f' % f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', dest='train')
    parser.add_argument("--eval", action='store_true', dest='eval')

    parser.add_argument("--train_file", action='store', dest="train_file")
    parser.add_argument("--test_file", action='store', dest="test_file")
    parser.add_argument("--word_vectors_file", action='store', dest="word_vectors_file")
    parser.add_argument("--model_file", action='store', dest="model_file")
    parser.add_argument("--gpus", nargs='+', action="store", dest="gpus")
    options = parser.parse_args()

    if options.gpus != None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(options.gpus)
        print("Working on following gpu cores: %s" % os.environ["CUDA_VISIBLE_DEVICES"])

    if options.train:
        # train and save model.
        all_data = generate_train_data(options.train_file, options.word_vectors_file)
        train_data = [all_data['train_X'], all_data['train_y']]
        dev_data = [all_data['dev_X'], all_data['dev_y']]
        train_model(train_data, dev_data, options.model_file)

        # performace on dev-set.
        models = load_model(options.model_file, load_hidden=True)
        eval(models, all_data['dev_X'], all_data['dev_y'])

    if options.eval:
        test_X, test_Y = generate_test_data(options.test_file, options.word_vectors_file)
        models = load_model(options.model_file, load_hidden=True)
        eval(models, test_X, test_Y)

