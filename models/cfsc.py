from collections import Counter
from glob import glob
from os.path import exists
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from time import time
from tqdm import tqdm
import argparse
import json
import logging
import numpy as np
import pandas as pd
import pickle
import pdb
import random
import sys

np.random.seed(865)
random.seed(865)

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

        if exists(self.features_path_trn) and exists(self.features_path_tst):
            self.logger.info('Features already computed')
            return

        self.logger.info('Reading dataframes')
        TRN = pd.read_csv('%s/train.csv' % self.data_dir)
        TST = pd.read_csv('%s/test.csv' % self.data_dir)
        SNG = pd.read_csv('%s/songs.csv' % self.data_dir)
        SNG = SNG[['song_id', 'song_length', 'artist_name']]

        self.logger.info('Merging train, test with songs')
        TRN = TRN.merge(SNG, on='song_id', how='left')
        TST = TST.merge(SNG, on='song_id', how='left')

        # Impute missing values with the most common value.
        impute_cols = ['artist_name']
        for c in impute_cols:
            self.logger.info('Imputing %s' % c)
            cmb = TRN[c].append(TST[c])
            val = cmb.value_counts().idxmax()
            TRN[c].fillna(val, inplace=True)
            TST[c].fillna(val, inplace=True)

        # Encode a subset of the columns.
        encode_cols = [
            ('msno', 'user_index'),
            ('song_id', 'song_index'),
            ('artist_name', 'artist_index'),
        ]
        for ca, cb in encode_cols:
            self.logger.info('Encoding %s -> %s' % (ca, cb))
            cmb = TRN[ca].append(TST[ca])
            enc = LabelEncoder()
            enc.fit(cmb)
            TRN[cb] = enc.transform(TRN[ca])
            TST[cb] = enc.transform(TST[ca])

        self.logger.info('Computing similar users for out-of-vocab users')
        ui_iv = TRN['user_index'].unique()
        ui_ov = np.setdiff1d(TST['user_index'], ui_iv)
        X = np.zeros((NB_USERS, NB_ARTISTS), dtype=np.uint8)
        C = TRN.append(TST)
        for ui, ai in C[['user_index', 'artist_index']].values:
            X[ui, ai] += 1

        knn = NearestNeighbors(n_neighbors=1, metric='cosine')
        knn.fit(csr_matrix(X[ui_iv]))
        nbs = ui_iv[knn.kneighbors(csr_matrix(X[ui_ov]), return_distance=False)[:, 0]]
        nbs_lookup = dict(zip(np.concatenate([ui_ov, ui_iv]), np.concatenate([nbs, ui_iv])))
        TST['user_index_'] = [nbs_lookup[ui] for ui in TST['user_index']]
        assert set(TST['user_index_']) - set(TRN['user_index']) == set([])

        self.logger.info('Removing unused columns')
        keep_cols_trn = ['user_index', 'song_index', 'artist_index', 'target']
        keep_cols_tst = ['id', 'user_index', 'song_index', 'artist_index', 'user_index_']
        TRN = TRN[keep_cols_trn]
        TST = TST[keep_cols_tst]

        self.logger.info('Saving dataframes')
        TRN.to_csv(self.features_path_trn, index=False)
        self.logger.info('Saved %s' % self.features_path_trn)
        TST.to_csv(self.features_path_tst, index=False)
        self.logger.info('Saved %s' % self.features_path_tst)

    def fit(self):

        net = self._networks(self.embedding_size)
        net.summary()
        net.compile(loss='binary_crossentropy',
                    optimizer=Adam(**self.embedding_optimizer_args),
                    metrics=['accuracy'])

        self.logger.info('%d users, %d songs' % (NB_USERS, NB_SONGS))

        cb = [
            ModelCheckpoint(self.embedding_path, monitor='loss', save_best_only=True, verbose=1, mode='min'),
            EarlyStopping(monitor='loss', patience=10, min_delta=0.002, verbose=1),
            ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, epsilon=0.005, min_lr=0.0001, verbose=1),
            CSVLogger('%s/cfsc_logs.csv' % self.artifacts_dir)
        ]

        TRN = pd.read_csv(self.features_path_trn)
        X = [TRN['user_index'], TRN['song_index']]
        Y = TRN['target']

        net.fit(X, Y,
                batch_size=self.embedding_batch,
                epochs=self.embedding_epochs,
                callbacks=cb)

    def predict(self):

        net = self._networks(self.embedding_size)
        net.load_weights(self.embedding_path, by_name=True)

        TST = pd.read_csv(self.features_path_tst)
        X = [TST['user_index_'], TST['song_index']]
        Y_prd = net.predict(X, batch_size=100000)

        SUB = pd.DataFrame({'id': TST['id'], 'target': Y_prd[:, 0]})
        SUB.to_csv(self.predict_path_tst, index=False)

        self.logger.info('Target mean %.3lf' % SUB['target'].mean())
        self.logger.info('Saved %s' % self.predict_path_tst)

    @staticmethod
    def _networks(embed_size, nb_users=NB_USERS, nb_songs=NB_SONGS):
        inp_u1 = Input((1,))
        inp_s1 = Input((1,))
        emb_uu = Embedding(nb_users, embed_size, embeddings_initializer=RandomNormal(0, 0.01))
        emb_ss = Embedding(nb_songs, embed_size, embeddings_initializer=RandomNormal(0, 0.01))
        emb_u1 = emb_uu(inp_u1)
        emb_s1 = emb_ss(inp_s1)
        emb_u1 = Reshape((embed_size,))(emb_u1)
        emb_s1 = Reshape((embed_size,))(emb_s1)
        dot_u1_s1 = dot([emb_u1, emb_s1], axes=-1)
        clf_u1_s1 = Activation('sigmoid', name='u1_s1')(dot_u1_s1)
        return Model([inp_u1, inp_s1], clf_u1_s1)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description='')
    ap.add_argument('--fit', action='store_true', default=False)
    ap.add_argument('--predict', action='store_true', default=False)
    args = vars(ap.parse_args())

    model = CFSC(
        data_dir='data',
        artifacts_dir='artifacts/cfsc',
        features_path_trn='artifacts/cfsc/features_trn.csv',
        features_path_tst='artifacts/cfsc/features_tst.csv',
        embedding_path='artifacts/cfsc/keras_embeddings_best.hdf5',
        predict_path_tst='artifacts/cfsc/predict_tst_%d.csv' % int(time()),
        embedding_size=100,
        embedding_epochs=20,
        embedding_batch=32000,
        embedding_optimizer_args={'lr': 0.001, 'decay': 1e-4}
    )

    model.get_features()

    if args['fit']:
        model.fit()

    if args['predict']:
        model.predict()
