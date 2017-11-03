from hashlib import md5
from math import ceil
from os import getenv
from os.path import exists
from scipy.misc import imread
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, Imputer
from time import time
from tqdm import tqdm
import argparse
import json
import logging
import numpy as np
import pandas as pd
import pdb
import sys

np.random.seed(865)

from keras.layers import Input, Embedding, Activation, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Dense, concatenate, dot, Lambda, Dropout, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import backend as K

MISSING_TOKEN = 'xxxxxxxx'


class AUC(Callback):

    def __init__(self, X, Y, nb_trn):
        self.X = X
        self.Y = Y
        self.nb_trn = nb_trn

    def on_epoch_end(self, epoch, logs={}):
        Yp = self.model.predict(self.X, batch_size=1000000)
        logs['auc'] = roc_auc_score(self.Y[:self.nb_trn], Yp[:self.nb_trn])
        logs['val_auc'] = roc_auc_score(self.Y[self.nb_trn:], Yp[self.nb_trn:])
        print('\n', logs)


class SpecVecRec(object):
    """Get SpecVecRec't"""

    def __init__(self,
                 data_dir,
                 artifacts_dir,
                 features_path_trn,
                 features_path_tst,
                 model_path,
                 predict_path_tst,
                 vec_size,
                 epochs,
                 batch,
                 optimizer_args):

        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        self.features_path_trn = features_path_trn
        self.features_path_tst = features_path_tst
        self.model_path = model_path
        self.predict_path_tst = predict_path_tst
        self.vec_size = vec_size
        self.epochs = epochs
        self.batch = batch
        self.optimizer_args = optimizer_args
        self.logger = logging.getLogger('VecRec')

    def get_features(self):
        """
        Current Features: 
        u_idx, u_cit, u_age, u_gen, 
        s_idx, s_art, s_lan, s_gen, s_yea, s_cou,
        target (train only)

        Potential Features: source_*, expiration dates
        """

        if exists(self.features_path_trn) and exists(self.features_path_tst):
            self.logger.info('Features already computed')
            return

        self.logger.info('Reading dataframes')
        TRN = pd.read_csv('%s/train.csv' % self.data_dir, usecols=['msno', 'song_id', 'target'])
        TST = pd.read_csv('%s/test.csv' % self.data_dir, usecols=['id', 'msno', 'song_id'])
        SNG = pd.read_csv('%s/songs.csv' % self.data_dir,
                          usecols=['song_id', 'artist_name', 'genre_ids', 'language'])
        SEI = pd.read_csv('%s/song_extra_info.csv' % self.data_dir, usecols=['song_id', 'isrc'])
        MMB = pd.read_csv('%s/members.csv' % self.data_dir, usecols=['msno', 'city', 'bd', 'gender'])

        self.logger.info('Merge SNG and SEI.')
        SNG = SNG.merge(SEI, on='song_id', how='left')

        self.logger.info('Merge TRN and TEST with SNG and MMB.')
        TRN = TRN.merge(MMB, on='msno', how='left')
        TST = TST.merge(MMB, on='msno', how='left')
        TRN = TRN.merge(SNG, on='song_id', how='left')
        TST = TST.merge(SNG, on='song_id', how='left')

        # Throw away unused after merging.
        del SEI
        del MMB

        # Impute with a common missing token.
        for c in TRN.columns[TRN.isnull().any()]:
            self.logger.info('Imputing %s' % c)
            TRN[c].fillna(MISSING_TOKEN, inplace=True)

        for c in TST.columns[TST.isnull().any()]:
            self.logger.info('Imputing %s' % c)
            TST[c].fillna(MISSING_TOKEN, inplace=True)

        assert len(TRN.columns[TRN.isnull().any()]) == 0
        assert len(TST.columns[TST.isnull().any()]) == 0

        self.logger.info('Collapsing out-of-vocab users')
        print("TODO")
        self.logger.info('Collapsing out-of-vocab songs')
        print("TODO")

        def round_year(n, s=5):
            try:
                return round(int(n) / s) * s
            except (TypeError, ValueError):
                return n

        self.logger.info('Converting ISRC to year and country')
        TRN['country'] = TRN['isrc'].apply(lambda s: s[:2])
        TST['country'] = TST['isrc'].apply(lambda s: s[:2])
        TRN['year'] = TRN['isrc'].apply(lambda s: round_year(s[5:7]))
        TST['year'] = TST['isrc'].apply(lambda s: round_year(s[5:7]))
        TRN.drop('isrc', axis=1, inplace=True)
        TST.drop('isrc', axis=1, inplace=True)

        self.logger.info('Normalizing genres')
        sort_genres = lambda s: '|'.join(sorted(s.split('|')))
        TRN['genre_ids'] = TRN['genre_ids'].apply(sort_genres)
        TST['genre_ids'] = TST['genre_ids'].apply(sort_genres)

        self.logger.info('Normalizing artists')
        sort_artists = lambda s: '|'.join(sorted(s.lower().split('| ')))
        TRN['artist_name'] = TRN['artist_name'].apply(sort_artists)
        TST['artist_name'] = TST['artist_name'].apply(sort_artists)

        self.logger.info('Normalizing age')
        round_age = lambda n, s=5: round(min(max(n, 0), 50) / s) * s
        TRN['bd'] = TRN['bd'].apply(round_age)
        TST['bd'] = TST['bd'].apply(round_age)

        cc = [
            ('msno', 'u_idx'),
            ('city', 'u_cit'),
            ('bd', 'u_age'),
            ('gender', 'u_gen'),
            ('song_id', 's_idx'),
            ('artist_name', 's_art'),
            ('language', 's_lan'),
            ('genre_ids', 's_gen'),
            ('year', 's_yea'),
            ('country', 's_cou'),
        ]

        for cold, cnew in cc:
            self.logger.info('Label encoding %s -> %s' % (cold, cnew))
            enc = LabelEncoder()
            enc.fit(TRN[cold].append(TST[cold]).apply(str))
            TRN[cnew] = enc.transform(TRN[cold].apply(str))
            TST[cnew] = enc.transform(TST[cold].apply(str))
            TRN.drop(cold, axis=1, inplace=True)
            TST.drop(cold, axis=1, inplace=True)
            assert cold not in TRN.columns
            assert cold not in TST.columns

        assert TRN.isnull().any().sum() == 0
        assert TST.isnull().any().sum() == 0

        self.logger.info('Saving dataframes')
        TRN.to_csv(self.features_path_trn, index=False)
        self.logger.info('Saved %s: (%dx%d)' % (self.features_path_trn, *TRN.shape))
        TST.to_csv(self.features_path_tst, index=False)
        self.logger.info('Saved %s: (%dx%d)' % (self.features_path_tst, *TST.shape))

    def network(self, nb_u_idx, nb_u_cit, nb_u_age, nb_u_gen, nb_s_idx, nb_s_art, nb_s_lan, nb_s_gen, nb_s_yea, nb_s_cou):
        """
        Inputs: 
        u_idx, u_cit, u_age, u_gen, 
        s_idx, s_art, s_lan, s_gen, s_yea, s_cou,

        Similarity interactions representing relationships between entities.

        u_idx, s_idx: user likes song?
        u_idx, s_art: user likes artist?
        u_idx, s_lan: user likes language?
        u_idx, s_gen: user likes genre?
        u_idx, s_yea: user likes year?
        u_idx, s_cou: user likes country?
        u_cit, s_idx: city likes song?
        u_cit, s_art: city likes artist?
        u_cit, s_lan: city likes language?
        u_cit, s_gen: city likes genre?
        u_cit, s_cou: city likes country?
        u_age, s_idx: age likes song?
        u_age, s_art: age likes artist?
        u_age, s_lan: age likes language?
        u_age, s_gen: age likes genre?
        u_age, s_yea: age likes year?
        u_gen, s_idx: gender likes song?
        u_gen, s_art: gender likes artist?
        u_gen, s_gen: gender likes genre?
        """

        vecs_u_idx = Embedding(nb_u_idx, self.vec_size, embeddings_initializer=RandomNormal(0, 0.01))
        vecs_u_cit = Embedding(nb_u_cit, self.vec_size, embeddings_initializer=RandomNormal(0, 0.01))
        vecs_u_age = Embedding(nb_u_age, self.vec_size, embeddings_initializer=RandomNormal(0, 0.01))
        vecs_u_gen = Embedding(nb_u_gen, self.vec_size, embeddings_initializer=RandomNormal(0, 0.01))
        vecs_s_idx = Embedding(nb_s_idx, self.vec_size, embeddings_initializer=RandomNormal(0, 0.01))
        vecs_s_art = Embedding(nb_s_art, self.vec_size, embeddings_initializer=RandomNormal(0, 0.01))
        vecs_s_lan = Embedding(nb_s_lan, self.vec_size, embeddings_initializer=RandomNormal(0, 0.01))
        vecs_s_gen = Embedding(nb_s_gen, self.vec_size, embeddings_initializer=RandomNormal(0, 0.01))
        vecs_s_yea = Embedding(nb_s_yea, self.vec_size, embeddings_initializer=RandomNormal(0, 0.01))
        vecs_s_cou = Embedding(nb_s_cou, self.vec_size, embeddings_initializer=RandomNormal(0, 0.01))

        inp_u_idx = Input(shape=(1,))
        inp_u_cit = Input(shape=(1,))
        inp_u_age = Input(shape=(1,))
        inp_u_gen = Input(shape=(1,))
        inp_s_idx = Input(shape=(1,))
        inp_s_art = Input(shape=(1,))
        inp_s_lan = Input(shape=(1,))
        inp_s_gen = Input(shape=(1,))
        inp_s_yea = Input(shape=(1,))
        inp_s_cou = Input(shape=(1,))

        vec_u_idx = Reshape((self.vec_size,))(vecs_u_idx(inp_u_idx))
        vec_u_cit = Reshape((self.vec_size,))(vecs_u_cit(inp_u_cit))
        vec_u_age = Reshape((self.vec_size,))(vecs_u_age(inp_u_age))
        vec_u_gen = Reshape((self.vec_size,))(vecs_u_gen(inp_u_gen))
        vec_s_idx = Reshape((self.vec_size,))(vecs_s_idx(inp_s_idx))
        vec_s_art = Reshape((self.vec_size,))(vecs_s_art(inp_s_art))
        vec_s_lan = Reshape((self.vec_size,))(vecs_s_lan(inp_s_lan))
        vec_s_gen = Reshape((self.vec_size,))(vecs_s_gen(inp_s_gen))
        vec_s_yea = Reshape((self.vec_size,))(vecs_s_yea(inp_s_yea))
        vec_s_cou = Reshape((self.vec_size,))(vecs_s_cou(inp_s_cou))

        cmb = concatenate([
            dot([vec_u_idx, vec_s_idx], axes=-1, normalize=True),  # user likes song?
            dot([vec_u_idx, vec_s_art], axes=-1, normalize=True),  # user likes artist?
            dot([vec_u_idx, vec_s_lan], axes=-1, normalize=True),  # user likes language?
            dot([vec_u_idx, vec_s_gen], axes=-1, normalize=True),  # user likes genre?
            dot([vec_u_idx, vec_s_yea], axes=-1, normalize=True),  # user likes year?
            dot([vec_u_idx, vec_s_cou], axes=-1, normalize=True),  # user likes country?
            dot([vec_u_cit, vec_s_idx], axes=-1, normalize=True),  # city likes song?
            dot([vec_u_cit, vec_s_art], axes=-1, normalize=True),  # city likes artist?
            dot([vec_u_cit, vec_s_lan], axes=-1, normalize=True),  # city likes language?
            dot([vec_u_cit, vec_s_gen], axes=-1, normalize=True),  # city likes genre?
            dot([vec_u_cit, vec_s_cou], axes=-1, normalize=True),  # city likes country?
            dot([vec_u_age, vec_s_idx], axes=-1, normalize=True),  # age likes song?
            dot([vec_u_age, vec_s_art], axes=-1, normalize=True),  # age likes artist?
            dot([vec_u_age, vec_s_lan], axes=-1, normalize=True),  # age likes language?
            dot([vec_u_age, vec_s_gen], axes=-1, normalize=True),  # age likes genre?
            dot([vec_u_age, vec_s_yea], axes=-1, normalize=True),  # age likes year?
            dot([vec_u_gen, vec_s_idx], axes=-1, normalize=True),  # gender likes song?
            dot([vec_u_gen, vec_s_art], axes=-1, normalize=True),  # gender likes artist?
            dot([vec_u_gen, vec_s_gen], axes=-1, normalize=True),  # gender likes genre?
        ])

        log = Dense(1)(cmb)
        clf = Activation('sigmoid')(log)
        return Model([inp_u_idx,
                      inp_u_cit,
                      inp_u_age,
                      inp_u_gen,
                      inp_s_idx,
                      inp_s_art,
                      inp_s_lan,
                      inp_s_gen,
                      inp_s_yea,
                      inp_s_cou], clf)

    def fit(self):

        # All data needed to compute numbers of embeddings.
        TRN = pd.read_csv(self.features_path_trn)
        TST = pd.read_csv(self.features_path_tst)
        CMB = TRN.append(TST)

        net = self.network(nb_u_idx=CMB['u_idx'].max() + 1,
                           nb_u_cit=CMB['u_cit'].max() + 1,
                           nb_u_age=CMB['u_age'].max() + 1,
                           nb_u_gen=CMB['u_gen'].max() + 1,
                           nb_s_idx=CMB['s_idx'].max() + 1,
                           nb_s_art=CMB['s_art'].max() + 1,
                           nb_s_lan=CMB['s_lan'].max() + 1,
                           nb_s_gen=CMB['s_gen'].max() + 1,
                           nb_s_yea=CMB['s_yea'].max() + 1,
                           nb_s_cou=CMB['s_cou'].max() + 1)
        net.summary()
        net.compile(loss='binary_crossentropy', optimizer=Adam(**self.optimizer_args), metrics=['accuracy'])

        # Split such that all users and songs in val have >= 1 record in trn.
        nb_trn = round(len(TRN) * 0.9)
        freq_users = TRN.groupby(['u_idx'])['u_idx'].transform('count').values
        freq_songs = TRN.groupby(['s_idx'])['s_idx'].transform('count').values
        cands_val, = np.where(((freq_users > 1) * (freq_songs > 1)) == True)
        ii_val = np.random.choice(cands_val, len(TRN) - nb_trn, replace=False)
        ii_trn = np.setdiff1d(np.arange(len(TRN)), ii_val)
        assert len(ii_trn) + len(ii_val) == len(TRN)

        # Abuse keras validation setup.
        TRN = TRN.iloc[list(ii_trn) + list(ii_val)]
        assert np.all(TRN.index == sorted(TRN.index)) == False

        X = [
            TRN['u_idx'].values,
            TRN['u_cit'].values,
            TRN['u_age'].values,
            TRN['u_gen'].values,
            TRN['s_idx'].values,
            TRN['s_art'].values,
            TRN['s_lan'].values,
            TRN['s_gen'].values,
            TRN['s_yea'].values,
            TRN['s_cou'].values,
        ]

        Y = TRN['target']

        cb = [
            AUC(X, Y, nb_trn),
            ModelCheckpoint(self.model_path, monitor='val_auc', save_best_only=True, mode='max', verbose=1),
            EarlyStopping(monitor='val_auc', patience=5, min_delta=0.001, mode='max', verbose=1),
            ReduceLROnPlateau(monitor='val_auc', factor=0.1, patience=3,
                              epsilon=1e-3, min_lr=1e-4, mode='max', verbose=1),
            CSVLogger('%s/vecrec_logs.csv' % self.artifacts_dir),
            TensorBoard(log_dir=self.artifacts_dir, histogram_freq=1, batch_size=100000, write_grads=True)
        ]

        net.fit(X, Y, validation_split=0.1, epochs=self.epochs, batch_size=self.batch, callbacks=cb)

    def predict(self):

        # All data needed to compute numbers of embeddings.
        TRN = pd.read_csv(self.features_path_trn)
        TST = pd.read_csv(self.features_path_tst)
        CMB = TRN.append(TST)

        net = self.network(nb_u_idx=CMB['u_idx'].max() + 1,
                           nb_u_cit=CMB['u_cit'].max() + 1,
                           nb_u_age=CMB['u_age'].max() + 1,
                           nb_u_gen=CMB['u_gen'].max() + 1,
                           nb_s_idx=CMB['s_idx'].max() + 1,
                           nb_s_art=CMB['s_art'].max() + 1,
                           nb_s_lan=CMB['s_lan'].max() + 1,
                           nb_s_gen=CMB['s_gen'].max() + 1,
                           nb_s_yea=CMB['s_yea'].max() + 1,
                           nb_s_cou=CMB['s_cou'].max() + 1)
        net.summary()

        X = [
            TST['u_idx'].values,
            TST['u_cit'].values,
            TST['u_age'].values,
            TST['u_gen'].values,
            TST['s_idx'].values,
            TST['s_art'].values,
            TST['s_lan'].values,
            TST['s_gen'].values,
            TST['s_yea'].values,
            TST['s_cou'].values,
        ]
        Yp = net.predict(X, batch_size=100000, verbose=True)

        SUB = pd.DataFrame({'id': TST['id'], 'target': Yp[:, 0]})
        SUB.to_csv(self.predict_path_tst, index=False)

        self.logger.info('Target mean %.3lf' % SUB['target'].mean())
        self.logger.info('Saved %s' % self.predict_path_tst)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description='')
    ap.add_argument('--fit', action='store_true', default=False)
    ap.add_argument('--predict', action='store_true', default=False)
    args = vars(ap.parse_args())

    model = SpecVecRec(
        data_dir='data',
        artifacts_dir='artifacts/catvecrec',
        features_path_trn='artifacts/catvecrec/features_trn.csv',
        features_path_tst='artifacts/catvecrec/features_tst.csv',
        model_path='artifacts/catvecrec/keras_embeddings_best.hdf5',
        predict_path_tst='artifacts/catvecrec/predict_tst_%d.csv' % int(time()),
        vec_size=60,
        epochs=15,
        batch=40000,
        optimizer_args={'lr': 0.001, 'decay': 1e-4}
    )

    model.get_features()

    if args['fit']:
        model.fit()

    elif args['predict']:
        model.predict()
