from collections import Counter
from glob import glob
from os.path import exists
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import argparse
import dill
import json
import logging
import numpy as np
import pandas as pd
import pdb

np.random.seed(865)

from keras.layers import Input, Embedding, Activation, dot, Reshape
from keras.models import Model, load_model
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback, EarlyStopping, ReduceLROnPlateau
from keras import backend as K

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

LAYER_NAME_USERS = 'EMBED_USERS'
LAYER_NAME_SONGS = 'EMBED_SONGS'
LAYER_NAME_ARTISTS = 'EMBED_ARTISTS'

NB_USERS = 34403
NB_SONGS = 419839
NB_ARTISTS = 46372


class CFSC(object):
    """
    Content-Free Similarity Classifier.
    Classifier that uses user, song, and artist similarities without
    explicitly processing any content."""

    def __init__(self,
                 data_dir,
                 artifacts_dir,
                 features_path_trn,
                 features_path_tst,
                 embedding_path,
                 predict_path_tst,
                 embedding_size,
                 embedding_epochs,
                 embedding_batch,
                 embedding_optimizer_args):

        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        self.features_path_trn = features_path_trn
        self.features_path_tst = features_path_tst
        self.embedding_path = embedding_path
        self.predict_path_tst = predict_path_tst
        self.embedding_size = embedding_size
        self.embedding_epochs = embedding_epochs
        self.embedding_batch = embedding_batch
        self.embedding_optimizer_args = embedding_optimizer_args
        self.logger = logging.getLogger('CFSC')

    def get_features(self):

        self.logger.info('Computing features')
        if exists(self.features_path_trn) and exists(self.features_path_tst):
            self.logger.info('Features already computed')
            return

        self.logger.info('Reading dataframes')
        TRN = pd.read_csv('%s/train.csv' % self.data_dir)
        TST = pd.read_csv('%s/test.csv' % self.data_dir)
        # Encode a subset of the columns.
        encode_cols = [
            ('msno', 'user_index'),
            ('song_id', 'song_index'),

        ]
        for ca, cb in encode_cols:
            self.logger.info('Encoding %s -> %s' % (ca, cb))
            cmb = TRN[ca].append(TST[ca])
            enc = LabelEncoder()
            enc.fit(cmb)
            TRN[cb] = pd.Series(enc.transform(TRN[ca]), dtype='category')
            TST[cb] = pd.Series(enc.transform(TST[ca]), dtype='category')

        # Add similarity columns that will be populated.
        TRN['sim_user_song'] = np.zeros(len(TRN))
        TST['sim_user_song'] = np.zeros(len(TST))

        # Keep a subset of all the columns.
        self.logger.info('Removing unused columns')
        keep_cols_trn = [
            'user_index',
            'song_index',
            'sim_user_song',
            'target'
        ]
        keep_cols_tst = ['id'] + keep_cols_trn[:-1]
        TRN = TRN[keep_cols_trn]
        TST = TST[keep_cols_tst]

        # TODO: Impute users who exist in the test set but not the traning set.
        # 3648 users accounting for 184018 records in the test set.

        # TODO: Impute songs that exist in the test set but not the training set.
        # 59873 songs accounting for 1494 records in the test set.

        self.logger.info('Saving dataframes')
        TRN.to_csv(self.features_path_trn, index=False)
        self.logger.info('Saved %s' % self.features_path_trn)
        TST.to_csv(self.features_path_tst, index=False)
        self.logger.info('Saved %s' % self.features_path_tst)

    def fit_embedding(self):

        net_clf, net_sim = self._networks(self.embedding_size)
        net_clf.summary()
        net_clf.compile(loss='binary_crossentropy',
                        optimizer=Adam(**self.embedding_optimizer_args),
                        metrics=['accuracy'])

        self.logger.info('%d users, %d songs' % (NB_USERS, NB_SONGS))

        TRN = pd.read_csv(self.features_path_trn)
        X = [TRN['user_index'], TRN['song_index']]
        Y = TRN['target']

        cb = [
            ModelCheckpoint(self.embedding_path,
                            monitor='loss',
                            save_best_only=True,
                            verbose=1,
                            mode='min'),
            EarlyStopping(monitor='loss',
                          patience=10,
                          min_delta=0.002,
                          verbose=1),
            ReduceLROnPlateau(monitor='loss',
                              factor=0.1,
                              patience=5,
                              epsilon=0.005,
                              min_lr=0.0001,
                              verbose=1),
            CSVLogger('%s/cfsc_logs.csv' % self.artifacts_dir)
        ]

        net_clf.fit(X, Y,
                    batch_size=self.embedding_batch,
                    epochs=self.embedding_epochs,
                    callbacks=cb,
                    verbose=1)

    def predict(self):

        net_clf, net_sim = self._networks(self.embedding_size)
        net_clf.load_weights(self.embedding_path, by_name=True)

        TST = pd.read_csv(self.features_path_tst)
        X = [TST['user_index'], TST['song_index']]
        Y_prd = net_clf.predict(X, batch_size=100000)

        SUB = pd.DataFrame({'id': TST['id'], 'target': Y_prd[:, 0]})
        SUB.to_csv(self.predict_path_tst, index=False)

        self.logger.info('Target mean %.3lf' % SUB['target'].mean())
        self.logger.info('Saved %s' % self.predict_path_tst)

    @staticmethod
    def _networks(embed_size, nb_users=NB_USERS, nb_songs=NB_SONGS):

        inp_user = Input(shape=(1,))
        inp_song = Input(shape=(1,))

        emb_users = Embedding(nb_users, embed_size, embeddings_initializer=RandomNormal(0, 0.01))
        emb_user = emb_users(inp_user)
        emb_user = Reshape((embed_size,))(emb_user)

        emb_songs = Embedding(nb_songs, embed_size, embeddings_initializer=RandomNormal(0, 0.01))
        emb_song = emb_songs(inp_song)
        emb_song = Reshape((embed_size,))(emb_song)

        dot_user_song = dot([emb_user, emb_song], axes=-1)
        clf_user_song = Activation('sigmoid', name='user_song')(dot_user_song)

        # First network used for training.
        net_clf = Model([inp_user, inp_song], clf_user_song)

        # Second network used to compute similarities.
        sim_user_song = dot([emb_user, emb_song], axes=-1, normalize=True)
        net_sim = Model([inp_user, inp_song], sim_user_song)

        return net_clf, net_sim


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description='')
    ap.add_argument('--fit_embedding', action='store_true', default=False)
    ap.add_argument('--predict', action='store_true', default=False)
    args = vars(ap.parse_args())

    model = CFSC(
        data_dir='data',
        artifacts_dir='artifacts/cfsc',
        features_path_trn='artifacts/cfsc/features_trn.csv',
        features_path_tst='artifacts/cfsc/features_tst.csv',
        embedding_path='artifacts/cfsc/keras_embeddings_best.hdf5',
        predict_path_tst='artifacts/cfsc/predict_tst.csv',
        embedding_size=64,
        embedding_epochs=20,
        embedding_batch=32000,
        embedding_optimizer_args={'lr': 0.001, 'decay': 1e-4}
    )

    model.get_features()

    if args['fit_embedding']:
        model.fit_embedding()

    if args['predict']:
        model.predict()
