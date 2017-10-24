########################################
## import packages
########################################

import datetime
import numpy as np
import os
import pandas as pd
import pdb

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Dense, Input, Embedding, Dropout, Activation, Reshape
from keras.layers.merge import concatenate, dot
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from keras.initializers import RandomUniform
from keras.optimizers import RMSprop, Adam, SGD
from keras.initializers import RandomNormal

import pickle

if not os.path.exists('artifacts/lystdo/lystdo.pkl'):

    train = pd.read_csv('./data/train.csv')
    uid = train.msno
    sid = train.song_id
    target = train.target

    test = pd.read_csv('./data/test.csv')
    id_test = test.id
    uid_test = test.msno
    sid_test = test.song_id

    usr_encoder = LabelEncoder()
    usr_encoder.fit(uid.append(uid_test))
    uid = usr_encoder.transform(uid)
    uid_test = usr_encoder.transform(uid_test)

    sid_encoder = LabelEncoder()
    sid_encoder.fit(sid.append(sid_test))
    sid = sid_encoder.transform(sid)
    sid_test = sid_encoder.transform(sid_test)

    u_cnt = int(max(uid.max(), uid_test.max()) + 1)
    s_cnt = int(max(sid.max(), sid_test.max()) + 1)

    perm = np.random.permutation(len(train))
    trn_cnt = int(len(train) * 0.85)
    uid_trn = uid[perm[:trn_cnt]]
    uid_val = uid[perm[trn_cnt:]]
    sid_trn = sid[perm[:trn_cnt]]
    sid_val = sid[perm[trn_cnt:]]
    target_trn = target[perm[:trn_cnt]]
    target_val = target[perm[trn_cnt:]]

    with open('artifacts/lystdo/lystdo.pkl', 'wb') as fp:
        d = (u_cnt, s_cnt, uid_trn, uid_val, sid_trn, sid_val, target_trn, target_val, uid_test, sid_test, id_test)
        pickle.dump(d, fp)

else:
    with open('artifacts/lystdo/lystdo.pkl', 'rb') as fp:
        d = pickle.load(fp)
        u_cnt, s_cnt, uid_trn, uid_val, sid_trn, sid_val, target_trn, target_val, uid_test, sid_test, id_test = d

########################################
# define the model
########################################


def get_model():
    user_embeddings = Embedding(u_cnt,
                                64,
                                embeddings_initializer=RandomNormal(0, 0.01),
                                input_length=1,
                                trainable=True)
    song_embeddings = Embedding(s_cnt,
                                64,
                                embeddings_initializer=RandomNormal(0, 0.01),
                                input_length=1,
                                trainable=True)

    uid_input = Input(shape=(1,), dtype='int32')
    embedded_usr = user_embeddings(uid_input)
    embedded_usr = Reshape((64,))(embedded_usr)

    sid_input = Input(shape=(1,), dtype='int32')
    embedded_song = song_embeddings(sid_input)
    embedded_song = Reshape((64,))(embedded_song)

    preds = dot([embedded_usr, embedded_song], axes=1)

    preds = Activation('sigmoid')(preds)

    model = Model(inputs=[uid_input, sid_input], outputs=preds)

    opt = RMSprop(lr=1e-3)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

    return model

########################################
# train the model
########################################


model = get_model()
model.summary()
early_stopping = EarlyStopping(monitor='val_acc', patience=5)
model_path = 'artifacts/lystdo/bst_model.h5'
model_checkpoint = ModelCheckpoint(model_path, save_best_only=True,
                                   save_weights_only=True)

hist = model.fit([uid_trn, sid_trn], target_trn,
                 validation_data=([uid_val, sid_val], target_val),
                 epochs=100, batch_size=32768, shuffle=True,
                 callbacks=[early_stopping, model_checkpoint])
model.load_weights(model_path)

preds_val = model.predict([uid_val, sid_val], batch_size=32768)
val_auc = roc_auc_score(target_val, preds_val)

########################################
# make the submission
########################################

preds_test = model.predict([uid_test, sid_test], batch_size=32768, verbose=1)
sub = pd.DataFrame({'id': id_test, 'target': preds_test.ravel()})
sub.to_csv('artifacts/lystdo/sub_%.5f.csv' % (val_auc), index=False)
