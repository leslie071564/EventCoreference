# joint model of anaphoricity and event coreference.
import keras
from keras.layers import Input, Dense, Concatenate, Subtract, Dropout, LSTM
from keras.models import Model

def get_model(silent=True, hidden_size=[100, 30], activation_fx='relu'):
    # lemma & PoS vector.
    Input_lemma_1 = Input(shape=(336,))
    Input_lemma_2 = Input(shape=(336,))
    Lemma_diff = Subtract()([Input_lemma_1, Input_lemma_2])

    Dense_lemma_layer = Dense(hidden_size[0], activation=activation_fx)
    Embed_lemma_1 = Dense_lemma_layer(Input_lemma_1)
    Embed_lemma_2 = Dense_lemma_layer(Input_lemma_2)
    Embed_diff = Dense_lemma_layer(Lemma_diff)

    # auxiliary features of coref model: context-diff, argument-binary, mention-distance.
    Input_context_diff = Input(shape=(300,))
    Embed_context_diff = Dense(hidden_size[0], activation=activation_fx)(Input_context_diff)
    Input_aux = Input(shape=(9,))

    # conext embedding of posterior mention.
    Input_context_2 = Input(shape=(None, 300))
    Dense_context_layer = LSTM(hidden_size[0], activation='relu')
    Embed_context_2 = Dense_context_layer(Input_context_2)

    # anaphoricity auxiliary features.
    Input_ana_aux = Input(shape=(1,))

    ## anaphoricity model
    Feat_concat_anaph = Concatenate(axis=-1)([Embed_lemma_2, Embed_context_2, Input_ana_aux])
    Feat_concat_anaph = Dropout(0.2)(Feat_concat_anaph)
    Feat_concat_anaph = Dense(hidden_size[1], activation='relu', name='anaph_hidden')(Feat_concat_anaph)
    output_anaph = Dense(1, activation='sigmoid', name='anaph')(Feat_concat_anaph)

    ## coref model
    Feat_concat = Concatenate(axis=-1)([Embed_lemma_1, Embed_lemma_2, Embed_context_2, Embed_diff, Embed_context_diff, Input_aux])
    #Feat_concat = Concatenate(axis=-1)([Embed_lemma_1, Embed_lemma_2, Embed_diff, Embed_context_diff, Feat_concat_anaph, Input_aux])
    #Feat_concat = Dense(hidden_size[0], activation=activation_fx)(Feat_concat)
    Feat_concat = Dense(hidden_size[1], activation=activation_fx, name='coref_hidden')(Feat_concat)
    output_coref = Dense(1, activation='sigmoid', name='coref')(Feat_concat)

    ## construct model
    inputs = [Input_lemma_1, Input_lemma_2, Input_context_diff, Input_context_2, Input_aux, Input_ana_aux]
    outputs = [output_coref, output_anaph]
    model = Model(inputs=inputs, outputs=outputs)

    return model

def load_model(model_file, silent=True):
    model = get_model(silent=True)
    model.load_weights(model_file)

    return model

