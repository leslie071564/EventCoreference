# pair-wise classifer for within-document event coreference.
import keras
from keras.layers import Input, Dense, Concatenate, Subtract, Dropout, LSTM
from keras.models import Model


def get_model(silent=False, hidden_size=[100, 30], activation_fx='relu'):
    # embed event lemma and PoS.
    Dense_lemma_layer = Dense(hidden_size[0], activation=activation_fx)

    Input_lemma_1 = Input(shape=(336,))
    Input_lemma_2 = Input(shape=(336,))
    Lemma_diff = Subtract()([Input_lemma_1, Input_lemma_2])

    Embed_lemma_1 = Dense_lemma_layer(Input_lemma_1)
    Embed_lemma_2 = Dense_lemma_layer(Input_lemma_2)
    Embed_diff = Dense_lemma_layer(Lemma_diff)

    # context
    Dense_context_layer = Dense(hidden_size[0], activation=activation_fx)

    Input_context_diff = Input(shape=(300,))
    Embed_context_diff = Dense_context_layer(Input_context_diff)

    # conbine different features.
    Input_aux = Input(shape=(9,))

    Feat_concat = Concatenate(axis=-1)([Embed_lemma_1, Embed_lemma_2, Embed_diff, Embed_context_diff, Input_aux])
    Feat_concat = Dense(hidden_size[1], activation=activation_fx, name='coref_hidden')(Feat_concat)
    output = Dense(1, activation='sigmoid')(Feat_concat)

    # construct model
    inputs = [Input_lemma_1, Input_lemma_2, Input_context_diff, Input_aux]
    model = Model(inputs=inputs, outputs=[output])

    if not silent:
        print(model.summary())
    return model

def load_model(model_file, load_hidden=False):
    model = get_model(silent=True)
    model.load_weights(model_file)

    if not load_hidden:
        return model

    hidden_model = Model(inputs=model.input, 
                         outputs=model.get_layer('coref_hidden').output)
    return model, hidden_model

