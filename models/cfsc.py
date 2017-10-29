from collections import Counter
from glob import glob
from os.path import exists
from sklearn.preprocessing import LabelEncoder
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
                 history_users_path,
                 history_songs_path,
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
        self.history_users_path = history_users_path
        self.history_songs_path = history_songs_path
        self.embedding_path = embedding_path
        self.predict_path_tst = predict_path_tst
        self.embedding_size = embedding_size
        self.embedding_epochs = embedding_epochs
        self.embedding_batch = embedding_batch
        self.embedding_optimizer_args = embedding_optimizer_args
        self.logger = logging.getLogger('CFSC')

    def get_features(self):

        if exists(self.features_path_trn) \
                and exists(self.features_path_tst) \
                and exists(self.history_users_path) \
                and exists(self.history_songs_path):
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

        # Keep a subset of all the columns.
        self.logger.info('Removing unused columns')
        keep_cols_trn = [
            'user_index',
            'song_index',
            'target'
        ]
        keep_cols_tst = ['id'] + keep_cols_trn[:-1]
        TRN = TRN[keep_cols_trn]
        TST = TST[keep_cols_tst]

        self.logger.info('Saving dataframes')
        TRN.to_csv(self.features_path_trn, index=False)
        self.logger.info('Saved %s' % self.features_path_trn)
        TST.to_csv(self.features_path_tst, index=False)
        self.logger.info('Saved %s' % self.features_path_tst)

        CMB = TRN.append(TST)

        self.logger.info('Building user -> song history')
        gby = CMB.groupby('user_index')
        history_users = {ui: list(gby.get_group(ui)['song_index'].values)
                         for ui in tqdm(CMB['user_index'].unique())}
        with open(self.history_users_path, 'wb') as fp:
            pickle.dump(history_users, fp)

        self.logger.info('Building song -> user history')
        gby = CMB.groupby('song_index')
        history_songs = {si: list(gby.get_group(si)['user_index'].values)
                         for si in tqdm(CMB['song_index'].unique())}
        with open(self.history_songs_path, 'wb') as fp:
            pickle.dump(history_songs, fp)

    def fit_sampler(self, TRN, TST, HU, HS):
        """

        """

        user_index_max = max(HU.keys())
        song_index_max = max(HS.keys())
        N = len(TRN) // self.embedding_batch * self.embedding_batch
        MAX_HISTORY_SIZE = 16020

        while True:

            # Vectorizeable variables: u1, u2, s2, s3, y1, y2, y3.
            # Non-vectorizable variables: u3, s3 (will have to sample).
            TRN = TRN.sample(frac=1.0)
            U1 = TRN['user_index'].values
            S1 = TRN['song_index'].values
            U2 = np.random.randint(0, user_index_max, N)
            U3 = np.random.randint(0, user_index_max, N)
            S2 = np.random.randint(0, song_index_max, N)
            S3 = np.random.randint(0, song_index_max, N)
            Y1 = TRN['target'].values
            Y2 = Y3 = np.arange(N) % 2
            RII = np.random.randint(0, 2 * MAX_HISTORY_SIZE, N)

            # Make batches.
            for ei in range(0, N, self.embedding_batch):

                X = [
                    U1[ei:ei + self.embedding_batch],
                    S1[ei:ei + self.embedding_batch],
                    U2[ei:ei + self.embedding_batch],
                    U3[ei:ei + self.embedding_batch],
                    S2[ei:ei + self.embedding_batch],
                    S3[ei:ei + self.embedding_batch],
                ]

                Y = [
                    Y1[ei:ei + self.embedding_batch],
                    Y2[ei:ei + self.embedding_batch],
                    Y3[ei:ei + self.embedding_batch],
                ]

                rii = RII[ei:ei + self.embedding_batch]

                # Sample, populate u3, s3 for this batch.
                for bi in range(self.embedding_batch):

                    u2, u3, s2, s3, ri = X[2][bi], X[3][bi], X[4][bi], X[5][bi], rii[bi]

                    # if y2 = 1, set u3 to ensure that u2 and u3 share a song.
                    # if y2 = 0, set y1 = u2 and u3 share a song.
                    # TODO: fast way to do ^^.
                    if Y[1][bi]:
                        si = HU[u2][ri % len(HU[u2])]
                        X[3][bi] = HS[si][ri % len(HS[si])]

                    # if y3 = 1, set s3 to ensure that s3 and s3 share a user.
                    # if y3 = 0, set y2 = s2 an s3 share a user.
                    # TODO: fast way to do ^^.
                    if Y[2][bi]:
                        ui = HS[s2][ri % len(HS[s2])]
                        X[5][bi] = HU[ui][ri % len(HU[ui])]

                yield X, Y

    def fit(self):

        net_trn, net_tst = self._networks(self.embedding_size)
        net_trn.summary()
        net_trn.compile(loss='binary_crossentropy',
                        optimizer=Adam(**self.embedding_optimizer_args),
                        metrics=['accuracy'],
                        loss_weights={'u1_s1': 0.5, 's2_s3': 0.25, 'u2_u3': 0.25})

        self.logger.info('%d users, %d songs' % (NB_USERS, NB_SONGS))

        cb = [
            ModelCheckpoint(self.embedding_path, monitor='loss', save_best_only=True, verbose=1, mode='min'),
            EarlyStopping(monitor='loss', patience=10, min_delta=0.002, verbose=1),
            ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, epsilon=0.005, min_lr=0.0001, verbose=1),
            CSVLogger('%s/cfsc_logs.csv' % self.artifacts_dir)
        ]

        TRN = pd.read_csv(self.features_path_trn)
        TST = pd.read_csv(self.features_path_tst)
        with open(self.history_users_path, 'rb') as fp:
            HU = pickle.load(fp)
        with open(self.history_songs_path, 'rb') as fp:
            HS = pickle.load(fp)

        gen_trn = self.fit_sampler(TRN, TST, HU, HS)
        net_trn.fit_generator(gen_trn,
                              steps_per_epoch=len(TRN) // self.embedding_batch,
                              epochs=self.embedding_epochs,
                              callbacks=cb)

    def predict(self):

        net_trn, net_tst = self._networks(self.embedding_size)
        net_tst.load_weights(self.embedding_path, by_name=True)

        TST = pd.read_csv(self.features_path_tst)
        X = [TST['user_index'], TST['song_index']]
        Y_prd = net_tst.predict(X, batch_size=100000)

        SUB = pd.DataFrame({'id': TST['id'], 'target': Y_prd[:, 0]})
        SUB.to_csv(self.predict_path_tst, index=False)

        self.logger.info('Target mean %.3lf' % SUB['target'].mean())
        self.logger.info('Saved %s' % self.predict_path_tst)

    @staticmethod
    def _networks(embed_size, nb_users=NB_USERS, nb_songs=NB_SONGS):

        inp_u1 = Input((1,))
        inp_s1 = Input((1,))
        inp_u2 = Input((1,))
        inp_u3 = Input((1,))
        inp_s2 = Input((1,))
        inp_s3 = Input((1,))

        emb_uu = Embedding(nb_users, embed_size, embeddings_initializer=RandomNormal(0, 0.01))
        emb_ss = Embedding(nb_songs, embed_size, embeddings_initializer=RandomNormal(0, 0.01))

        emb_u1 = emb_uu(inp_u1)
        emb_u2 = emb_uu(inp_u2)
        emb_u3 = emb_uu(inp_u3)
        emb_s1 = emb_ss(inp_s1)
        emb_s2 = emb_ss(inp_s2)
        emb_s3 = emb_ss(inp_s3)

        emb_u1 = Reshape((embed_size,))(emb_u1)
        emb_u2 = Reshape((embed_size,))(emb_u2)
        emb_u3 = Reshape((embed_size,))(emb_u3)
        emb_s1 = Reshape((embed_size,))(emb_s1)
        emb_s2 = Reshape((embed_size,))(emb_s2)
        emb_s3 = Reshape((embed_size,))(emb_s3)

        dot_u1_s1 = dot([emb_u1, emb_s1], axes=-1)
        dot_u2_u3 = dot([emb_u2, emb_u3], axes=-1)
        dot_s2_s3 = dot([emb_s2, emb_s3], axes=-1)

        clf_u1_s1 = Activation('sigmoid', name='u1_s1')(dot_u1_s1)
        clf_u2_u3 = Activation('sigmoid', name='u2_u3')(dot_u2_u3)
        clf_s2_s3 = Activation('sigmoid', name='s2_s3')(dot_s2_s3)

        net_trn = Model([inp_u1, inp_s1, inp_u2, inp_u3, inp_s2, inp_s3],
                        [clf_u1_s1, clf_u2_u3, clf_s2_s3])
        net_tst = Model([inp_u1, inp_s1], [clf_u1_s1])
        return net_trn, net_tst


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
        history_users_path='artifacts/cfsc/history_users.pkl',
        history_songs_path='artifacts/cfsc/history_songs.pkl',
        embedding_path='artifacts/cfsc/keras_embeddings_best.hdf5',
        predict_path_tst='artifacts/cfsc/predict_tst_%d.csv' % int(time()),
        embedding_size=100,
        embedding_epochs=30,
        embedding_batch=32000,
        embedding_optimizer_args={'lr': 0.01, 'decay': 1e-4}
    )

    model.get_features()

    if args['fit']:
        model.fit()

    if args['predict']:
        model.predict()
