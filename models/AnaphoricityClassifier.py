import keras
from keras.layers import Input, Dense, Concatenate, Subtract, Dropout, LSTM
from keras.models import Model

def get_model(silent=True):
    # compress lemma & pos embedding.
    Input_lemma = Input(shape=(336,))
    Dense_lemma_layer = Dense(100, activation='relu')
    Embed_lemma = Dense_lemma_layer(Input_lemma)

    # context embedding with LSTM
    Input_context = Input(shape=(None, 300))
    Dense_context_layer = LSTM(100, activation='relu')
    Embed_context = Dense_context_layer(Input_context)

    # auxiliary features
    Input_aux = Input(shape=(1,))

    # concatenate to make final decision
    Feat_concat = Concatenate(axis=-1)([Embed_lemma, Embed_context, Input_aux])
    Feat_concat = Dense(30, activation='relu', name='anaph_hidden')(Feat_concat)
    output = Dense(1, activation='sigmoid')(Feat_concat)

    # create model
    inputs = [Input_lemma, Input_context, Input_aux]
    model = Model(inputs=inputs, outputs=[output])

    return model

def load_model(model_file, load_hidden=False):
    model = get_model(silent=True)
    model.load_weights(model_file)

    if not load_hidden:
        return model

    hidden_model = Model(inputs=model.input, 
                         outputs=model.get_layer('anaph_hidden').output)
    return model, hidden_model

