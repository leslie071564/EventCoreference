import os
import argparse
import pickle
import jsonlines
import keras
import numpy as np
from random import shuffle
from Document import Document
from EmbedMap import EmbedMap
from models.JointModel import get_model, load_model
from utils import convert_train_data

def generate_train_data(train_data_file, word_vectors_file, dev_ratio=50):
    # load word vectors and PoS one-hot vectors.
    embed_map = EmbedMap(word_vectors_file)

    # load training data file.
    print('loading training data from %s' % train_data_file)
    train_reader = jsonlines.open(train_data_file)

    # preprocess the training data and generate training instances.
    train_data, dev_data = [], []
    for ii, doc_data in enumerate(train_reader.iter()):
        doc = Document(doc_data, embed_map)
        if ii % dev_ratio == 0:
            #dev_data += doc.generate_gold_mention_pairs()
            #dev_data += doc.generate_candidate_mention_pairs(with_ref_index=False)
            dev_data += doc.generate_dev_mention_pairs()
        else:
            train_data += doc.generate_gold_mention_pairs()

    print("---> total number of training pairs: %s" % len(train_data))
    print("---> total number of development pairs: %s" % len(dev_data))

    # shuffle the training data and convert to seperate numpy arrays.
    train_X, train_y = convert_train_data(train_data, y_len=2)
    dev_X, dev_y = convert_train_data(dev_data, y_len=2)

    return {'train_X': train_X, 'train_y': train_y,
            'dev_X': dev_X, 'dev_y': dev_y}

def train_model(train_data, dev_data, model_file, lr=0.001, batch_size=512, epochs=20):
    # load train and dev data.
    train_X, train_y = train_data
    dev_X, dev_y = dev_data

    # load model
    print('load model')
    model = get_model(silent=True)

    # compile model
    print('---> complile model (learn-rate: %s)' % lr)
    opt = keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy', loss_weights=[1., 0.1],
                  metrics=['accuracy'])

    # train model and save the best model to file.
    print('---> train model (#mb: %s, #epoch: %s)' % (batch_size, epochs))
    checkpoint = keras.callbacks.ModelCheckpoint(model_file, monitor='val_coref_loss', verbose=1, save_best_only=True, mode='min')
    earlystop = keras.callbacks.EarlyStopping(monitor='val_coref_loss', min_delta=0, patience=2, verbose=0, mode='auto')
    class_weight = {'coref': {0:2., 1:10.}}
    hist = model.fit(train_X, train_y, validation_data=(dev_X, dev_y),
              batch_size=batch_size, epochs=epochs, shuffle=True,
              verbose=1, callbacks=[checkpoint, earlystop], class_weight=class_weight
              )

    # save loss and evaluation for ploting training graph
    print('trained model saved to: %s' % model_file)

def eval(model, x, gold_Y):
    # make predictions
    gold_Y = gold_Y[0]
    predict_Y = np.round(model.predict(x)[0]).reshape(gold_Y.shape)

    # statistics of test data.
    pos_test = np.count_nonzero(gold_Y)
    print("---> %s positive examples out of %s examples (%.2f%%)" % (pos_test, gold_Y.shape[0],
                                                                  (1- pos_test/gold_Y.shape[0]) * 100))

    # calculate precision, recall, F1 scores.
    correct_predict = np.count_nonzero(predict_Y * gold_Y)
    gold_coref_num = np.count_nonzero(gold_Y)
    predict_coref_num = np.count_nonzero(predict_Y)

    prec = correct_predict / predict_coref_num
    recall = correct_predict / gold_coref_num
    f1 = 2 * prec * recall / (prec + recall)

    # print evaluation scores.
    print("---> <positive pair> precision: %.4f, recall: %.4f, f1: %.4f" % (prec, recall, f1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs='+', action="store", dest="gpus")
    parser.add_argument("--train_file", action='store', dest="train_file")
    parser.add_argument("--word_vectors_file", action='store', dest="word_vectors_file")
    parser.add_argument("--model_file", action='store', dest="model_file")
    options = parser.parse_args()

    if options.gpus != None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(options.gpus)
        print("Working on following gpu cores: %s" % os.environ["CUDA_VISIBLE_DEVICES"])

    all_data = generate_train_data(options.train_file, options.word_vectors_file)
    train_data = [all_data['train_X'], all_data['train_y']]
    dev_data = [all_data['dev_X'], all_data['dev_y']]
    train_model(train_data, dev_data, options.model_file)

    print('evaluation on dev-set:')
    model = load_model(options.model_file)
    eval(model, all_data['dev_X'], all_data['dev_y'])

