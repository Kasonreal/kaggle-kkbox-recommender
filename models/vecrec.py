from collections import Counter
from glob import glob
from os.path import exists
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
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

from keras.layers import Input, Embedding, Activation, dot, Reshape
from keras.models import Model, load_model
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import backend as K

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

LAYER_NAME_USERS = 'EMBED_USERS'
LAYER_NAME_SONGS = 'EMBED_SONGS'

NB_USERS = 34403
NB_SONGS = 419839
NB_ARTISTS = 46372


class AUC(Callback):

    def __init__(self, Xt, Yt, Xv, Yv):
        self.Xt = Xt
        self.Yt = Yt
        self.Xv = Xv
        self.Yv = Yv

    def on_epoch_end(self, epoch, logs={}):
        Yp = self.model.predict(self.Xt, batch_size=1000000)
        logs['auc'] = roc_auc_score(self.Yt, Yp)
        Yp = self.model.predict(self.Xv, batch_size=1000000)
        logs['val_auc'] = roc_auc_score(self.Yv, Yp)
        print('\n', logs)


class VecDiffs(Callback):

    def __init__(self):
        self.layers = [LAYER_NAME_USERS, LAYER_NAME_SONGS]

    def on_train_begin(self, logs):
        self.W0 = [(l, self.model.get_layer(l).get_weights()) for l in self.layers]

    def on_epoch_end(self, epoch, logs):
        for l, w0 in self.W0:
            w1 = self.model.get_layer(l).get_weights()
            same = np.sum(np.all(w0[0] == w1[0], axis=1))
            print('%s: %d random vectors (%.3lf)' % (l, same, same / w0[0].shape[0]))


class VecRec(object):
    """Get VecRec't"""

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
        self.logger = logging.getLogger('VecRec')

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
        for ui, ai in TRN.append(TST)[['user_index', 'artist_index']].values:
            X[ui, ai] += 1

        knn = NearestNeighbors(n_neighbors=1, metric='cosine')
        knn.fit(csr_matrix(X[ui_iv]))
        nbs = ui_iv[knn.kneighbors(csr_matrix(X[ui_ov]), return_distance=False)[:, 0]]
        nbs = dict(zip(np.concatenate([ui_ov, ui_iv]), np.concatenate([nbs, ui_iv])))
        TST['user_index_'] = [nbs[ui] for ui in TST['user_index']]
        assert set(TST['user_index_']) - set(TRN['user_index']) == set([])

        self.logger.info('Removing unused columns')
        TRN = TRN[['user_index', 'song_index', 'artist_index', 'target']]
        TST = TST[['id', 'user_index', 'song_index', 'artist_index', 'user_index_']]

        self.logger.info('Saving dataframes')
        TRN.to_csv(self.features_path_trn, index=False)
        self.logger.info('Saved %s' % self.features_path_trn)
        TST.to_csv(self.features_path_tst, index=False)
        self.logger.info('Saved %s' % self.features_path_tst)

    @staticmethod
    def sgns(embed_size):
        embs_u = Embedding(NB_USERS, embed_size, name=LAYER_NAME_USERS, embeddings_initializer=RandomNormal(0, 0.01), embeddings_regularizer=l2(1e-5))
        embs_s = Embedding(NB_SONGS, embed_size, name=LAYER_NAME_SONGS, embeddings_initializer=RandomNormal(0, 0.01), embeddings_regularizer=l2(1e-5))
        inp_u, inp_s = Input((1,)), Input((1,))
        emb_u, emb_s = embs_u(inp_u), embs_s(inp_s)
        emb_u, emb_s = Reshape((embed_size,))(emb_u), Reshape((embed_size,))(emb_s)
        dot_u_s = dot([emb_u, emb_s], axes=-1)
        prb_u_s = Activation('sigmoid', name='u1_s1')(dot_u_s)
        return Model([inp_u, inp_s], prb_u_s)

    def fit(self):

        net = self.sgns(self.embedding_size)
        net.summary()
        net.compile(loss='binary_crossentropy',
                    optimizer=Adam(**self.embedding_optimizer_args),
                    metrics=['accuracy'])

        self.logger.info('%d users, %d songs' % (NB_USERS, NB_SONGS))

        # Load all training data.
        TRN = pd.read_csv(self.features_path_trn)

        # Split such that all users and songs in val have >= 1 record in trn.
        nb_trn = round(len(TRN) * 0.9)
        freq_users = TRN.groupby(['user_index'])['user_index'].transform('count').values
        freq_songs = TRN.groupby(['song_index'])['song_index'].transform('count').values
        cands_val, = np.where(((freq_users > 1) * (freq_songs > 1)) == True)
        ii_val = np.random.choice(cands_val, len(TRN) - nb_trn, replace=False)
        ii_trn = np.setdiff1d(np.arange(len(TRN)), ii_val)
        assert len(ii_trn) + len(ii_val) == len(TRN)

        VAL = TRN.iloc[ii_val]
        TRN = TRN.iloc[ii_trn]
        assert len(np.intersect1d(TRN.index, VAL.index)) == 0

        Xt, Yt = [TRN['user_index'], TRN['song_index']], TRN['target']
        Xv, Yv = [VAL['user_index'], VAL['song_index']], VAL['target']

        cb = [
            AUC(Xt, Yt, Xv, Yv),
            VecDiffs(),
            ModelCheckpoint(self.embedding_path, monitor='val_auc', save_best_only=True, verbose=1, mode='max'),
            EarlyStopping(monitor='val_auc', patience=20, min_delta=0.001, verbose=1, mode='max'),
            ReduceLROnPlateau(monitor='val_auc', factor=0.1, patience=10, epsilon=0.001, min_lr=0.0001, verbose=1),
            CSVLogger('%s/vecrec_logs.csv' % self.artifacts_dir),
            # TensorBoard(log_dir=self.artifacts_dir, histogram_freq=1, batch_size=100000, write_grads=True,
            #             embeddings_freq=1, embeddings_layer_names=[LAYER_NAME_USERS, LAYER_NAME_SONGS])
        ]

        net.fit(Xt, Yt,
                validation_data=(Xv, Yv),
                batch_size=self.embedding_batch,
                epochs=self.embedding_epochs,
                callbacks=cb)

    def predict(self):

        net = self.sgns(self.embedding_size)
        net.load_weights(self.embedding_path, by_name=True)

        TST = pd.read_csv(self.features_path_tst)
        X = [TST['user_index_'], TST['song_index']]
        Y_prd = net.predict(X, batch_size=100000)

        SUB = pd.DataFrame({'id': TST['id'], 'target': Y_prd[:, 0]})
        SUB.to_csv(self.predict_path_tst, index=False)

        self.logger.info('Target mean %.3lf' % SUB['target'].mean())
        self.logger.info('Saved %s' % self.predict_path_tst)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description='')
    ap.add_argument('--fit', action='store_true', default=False)
    ap.add_argument('--predict', action='store_true', default=False)
    args = vars(ap.parse_args())

    model = VecRec(
        data_dir='data',
        artifacts_dir='artifacts/vecrec',
        features_path_trn='artifacts/vecrec/features_trn.csv',
        features_path_tst='artifacts/vecrec/features_tst.csv',
        embedding_path='artifacts/vecrec/keras_embeddings_best.hdf5',
        predict_path_tst='artifacts/vecrec/predict_tst_%d.csv' % int(time()),
        embedding_size=100,
        embedding_epochs=100,
        embedding_batch=40000,
        embedding_optimizer_args={'lr': 0.001, 'decay': 1e-4}
    )

    model.get_features()

    if args['fit']:
        model.fit()

    if args['predict']:
        model.predict()
