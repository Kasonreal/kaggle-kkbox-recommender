from hashlib import md5
from math import ceil
from os import getenv
from os.path import exists
from scipy.misc import imread
from sklearn.preprocessing import LabelEncoder
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

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from models import make_parallel


LAYER_NAME_USERS = 'EMBED_USERS'
LAYER_NAME_SONGS = 'EMBED_SONGS'

NB_USERS = 34403
NB_SONGS = 419839
NB_ARTISTS = 46372

SPEC_MEAN = 115
SPEC_STDV = 48

NB_GPUS = getenv('CUDA_VISIBLE_DEVICES').count(',') + 1


class SpecReader(object):
    """Image reader with a cache for frequent images. aka cache money."""

    def __init__(self, spec_time, spec_freq, cache_paths, artifacts_dir, dtype=np.uint8):
        self.spec_time = spec_time
        self.spec_freq = spec_freq
        self.dtype = dtype
        self.path_to_index = {}
        logger = logging.getLogger(self.__class__.__name__)
        logger.info('Building cache with size %d' % len(cache_paths))
        self.cache = self.read(cache_paths, verbose=True)
        self.path_to_index = {p: i for i, p in enumerate(cache_paths)}

    def read(self, paths, verbose=False):
        specs = np.zeros((len(paths), self.spec_time, self.spec_freq), dtype=self.dtype)
        iterator = tqdm(enumerate(paths)) if verbose else enumerate(paths)
        for i, path in iterator:
            if path in self.path_to_index:
                specs[i] = self.cache[self.path_to_index[path]]
            else:
                specs[i] = self._read_single(path)
        return specs

    def _read_single(self, path):
        """Read from the center of the image. Tile the image if it is too small."""
        im = imread(path)
        if im.shape[1] < self.spec_time:
            im = np.tile(im, ceil(self.spec_time / im.shape[1]))
        cx, wx = im.shape[1] // 2, self.spec_time // 2
        return im[:self.spec_freq, cx - wx:cx + wx].T


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
                 spec_time,
                 spec_freq,
                 epochs,
                 batch,
                 optimizer_args,
                 cache_size):

        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        self.features_path_trn = features_path_trn
        self.features_path_tst = features_path_tst
        self.model_path = model_path
        self.predict_path_tst = predict_path_tst
        self.vec_size = vec_size
        self.spec_time = spec_time
        self.spec_freq = spec_freq
        self.epochs = epochs
        self.batch = batch
        self.optimizer_args = optimizer_args
        self.cache_size = cache_size
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

        # Subset of songs that is actually used, including those from the training
        # and testing tables which are not in the songs table.
        SNG_ = SNG[SNG['song_id'].isin(TRN['song_id'].append(TST['song_id']))]
        SNG_ = SNG_.append(TRN[~TRN['song_id'].isin(SNG_['song_id'])])
        SNG_ = SNG_.append(TST[~TST['song_id'].isin(SNG_['song_id'])])

        self.logger.info('Computing spectrogram paths')
        song_id_to_path = {}
        artist_to_paths = {x: [] for x in SNG['artist_name'].unique()}
        check_again = []
        for i, row in tqdm(SNG_.iterrows()):
            hash_ = sha256(row['song_id'].encode()).hexdigest()
            path_ = 'data/kkbox-melspecs/%s_melspec.jpg' % hash_
            if exists(path_):
                song_id_to_path[row['song_id']] = path_
                artist_to_paths[row['artist_name']].append(path_)
            else:
                check_again.append(row)

        self.logger.info('Imputing missing spectrograms for %d records' % len(check_again))
        for row in tqdm(check_again):
            if row['artist_name'] in artist_to_paths and len(artist_to_paths[row['artist_name']]) > 0:
                song_id_to_path[row['song_id']] = artist_to_paths[row['artist_name']][0]
            else:
                song_id_to_path[row['song_id']] = artist_to_paths['Various Artists'][0]

        TRN['spec_path'] = [song_id_to_path[x] for x in TRN['song_id']]
        TST['spec_path'] = [song_id_to_path[x] for x in TST['song_id']]

        self.logger.info('Removing unused columns')
        TRN = TRN[['user_index', 'song_index', 'spec_path', 'artist_index', 'target']]
        TST = TST[['id', 'user_index', 'song_index', 'spec_path', 'artist_index']]

        self.logger.info('Saving dataframes')
        TRN.to_csv(self.features_path_trn, index=False)
        self.logger.info('Saved %s' % self.features_path_trn)
        TST.to_csv(self.features_path_tst, index=False)
        self.logger.info('Saved %s' % self.features_path_tst)

    def network(self):

        # Input for user ID and song spectrogram.
        inp_user = Input(shape=(1,))
        inp_song = Input(shape=(self.spec_time, self.spec_freq))

        # Build the convolutional layers, based on Deep Content-based Music Recommendation (2014).
        x = Lambda(lambda x: (x - SPEC_MEAN) / SPEC_STDV)(inp_song)
        x = Dropout(0.2)(x)
        x = Conv1D(256, 4, strides=1, kernel_initializer='he_normal')(x)
        x = MaxPooling1D(4)(x)
        x = LeakyReLU()(x)

        x = Dropout(0.2)(x)
        x = Conv1D(256, 4, strides=1, kernel_initializer='he_normal')(x)
        x = MaxPooling1D(2)(x)
        x = LeakyReLU()(x)

        x = Dropout(0.2)(x)
        x = Conv1D(512, 4, strides=1, kernel_initializer='he_normal')(x)
        x = MaxPooling1D(2)(x)
        x = LeakyReLU()(x)

        global_max = GlobalMaxPooling1D()(x)
        global_avg = GlobalAveragePooling1D()(x)
        x = concatenate([global_max, global_avg])

        x = Dropout(0.2)(x)
        x = Dense(512)(x)
        x = LeakyReLU()(x)
        x = emb_song = out_song = Dense(self.vec_size)(x)

        net_song = Model(inp_song, out_song)

        # User-song combined embeddings network.
        emb_users = Embedding(NB_USERS, self.vec_size, name=LAYER_NAME_USERS,
                              embeddings_initializer=RandomNormal(0, 0.01))
        emb_user = emb_users(inp_user)
        emb_user = Reshape((self.vec_size,))(emb_user)
        dot_user_song = dot([emb_user, emb_song], axes=-1, name='dot')
        classify = Activation('sigmoid', name='classify')(dot_user_song)
        net_user_song = Model([inp_user, inp_song], classify)

        return net_user_song, net_song

    def batch_gen(self, steps, user_indexes, spec_paths, spec_reader, targets):

        while True:
            I = np.array_split(np.concatenate([
                np.random.permutation(len(user_indexes)),
                np.random.randint(0, len(user_indexes), steps * self.batch - len(user_indexes))
            ]), steps)
            for ii in I:
                specs_ = spec_reader.read(spec_paths[ii])
                yield [user_indexes[ii], specs_], targets[ii]

    def fit(self):

        net, _ = self.network()
        net.summary()
        if NB_GPUS > 1:
            net = make_parallel(net, NB_GPUS)
        net.compile(loss='binary_crossentropy', optimizer=Adam(**self.optimizer_args), metrics=['accuracy'])

        # Load all training data.
        TRN = pd.read_csv(self.features_path_trn)

        # Image reader that caches the most frequent images.
        paths = TRN['spec_path'].value_counts().index.values[:self.cache_size]
        spec_reader = SpecReader(self.spec_time, self.spec_freq, paths, self.artifacts_dir)

        # Split such that all users in val have >= 1 record in trn.
        nb_trn = round(len(TRN) * 0.9)
        freq_users = TRN.groupby(['user_index'])['user_index'].transform('count').values
        cands_val, = np.where(freq_users > 1)
        ii_val = np.random.choice(cands_val, len(TRN) - nb_trn, replace=False)
        ii_trn = np.setdiff1d(np.arange(len(TRN)), ii_val)
        assert len(ii_trn) + len(ii_val) == len(TRN)

        VAL = TRN.iloc[ii_val]
        TRN = TRN.iloc[ii_trn]
        assert len(np.intersect1d(TRN.index, VAL.index)) == 0

        # Generators for training and validation.
        steps_trn = (len(TRN) // self.batch * self.batch + self.batch) // self.batch
        steps_val = (len(VAL) // self.batch * self.batch + self.batch) // self.batch
        gen_trn = self.batch_gen(steps_trn, TRN['user_index'].values, TRN['spec_path'].values,
                                 spec_reader, TRN['target'].values)
        gen_val = self.batch_gen(steps_val, VAL['user_index'].values, VAL['spec_path'].values,
                                 spec_reader, VAL['target'].values)

        cb = [
            ModelCheckpoint(self.model_path, monitor='val_auc', save_best_only=True, verbose=1, mode='max'),
            EarlyStopping(monitor='val_auc', patience=20, min_delta=0.001, verbose=1, mode='max'),
            ReduceLROnPlateau(monitor='val_auc', factor=0.1, patience=10, epsilon=0.001, min_lr=0.0001, verbose=1),
            CSVLogger('%s/vecrec_logs.csv' % self.artifacts_dir),
        ]

        net.fit_generator(gen_trn, steps_per_epoch=steps_trn, validation_data=gen_val,
                          validation_steps=steps_val, callbacks=cb)

    def predict(self):

        net = self._networks(self.vec_size)
        net.load_weights(self.model_path, by_name=True)

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

    model = SpecVecRec(
        data_dir='data',
        artifacts_dir='artifacts/specvecrec',
        features_path_trn='artifacts/specvecrec/features_trn.csv',
        features_path_tst='artifacts/specvecrec/features_tst.csv',
        model_path='artifacts/specvecrec/keras_embeddings_best.hdf5',
        predict_path_tst='artifacts/specvecrec/predict_tst_%d.csv' % int(time()),
        vec_size=64,
        spec_time=500,
        spec_freq=128,
        epochs=100,
        batch=1700,
        optimizer_args={'lr': 0.001, 'decay': 1e-4},
        cache_size=95000
    )

    model.get_features()

    if args['fit']:
        model.fit()

    if args['predict']:
        model.predict()
