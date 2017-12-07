from collections import Counter
from hashlib import md5
from math import log, ceil
from more_itertools import flatten, chunked
from multiprocessing import Pool, cpu_count
from os.path import exists
from os import getenv
from pprint import pformat
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, ShuffleSplit
from time import time
from tqdm import tqdm
import argparse
import json
import logging
import numpy as np
import pandas as pd
import pdb
import re
import tensorflow as tf

NTRN = 7377418
NTST = 2556790
np.random.seed(865)

assert getenv('CUDA_VISIBLE_DEVICES') is not None, "Specify a GPU"
assert len(getenv('CUDA_VISIBLE_DEVICES')) > 0, "Specify a GPU"

from keras.engine.topology import Layer
from keras.layers import Input, Embedding, Activation, Lambda, concatenate, multiply, dot
from keras.models import Model, load_model
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import keras.backend as K

IAFM_HYPERPARAMS_DEFAULT = {
    'vec_size': 10,
    'vecs_init_func': np.random.normal,
    'vecs_init_kwargs': {'loc': 0, 'scale': 0.01},
    'vecs_reg_func': l2,
    'vecs_reg_kwargs': {'l': 0.0},
    'dropout_prop': 0.0,
    'optimizer': Adam,
    'optimizer_kwargs': {'lr': 0.01},
    'nb_epochs_max': 5,
    'batch_size': 50000,
    'early_stop_metric': 'val_auc_roc',
    'early_stop_delta': 0.005,
    'early_stop_patience': 2,
}


class VecReviewCallback(Callback):

    def __init__(self, vi2feat, nb_samples=6):
        self.vi2feat = vi2feat
        self.nb_samples = nb_samples
        self.logger = logging.getLogger(self.__class__.__name__)

    def on_epoch_end(self, epoch, logs):
        print('\n')

        # Extract the vectors.
        vecs = self.model.get_layer('vecs').get_weights()[0]

        # Make sure the padding vector has not been changed.
        assert np.all(vecs[0] == np.zeros_like(vecs[0])), 'Vector 0 must be all-zero'

        # Compute and sort vector norms. Print those with highest and lowest norm.
        norms = np.sqrt(np.sum(vecs ** 2, 1))
        norms_vi = np.argsort(norms)
        for i in reversed(norms_vi[-self.nb_samples:]):
            self.logger.info('%-60s %.3lf' % (self.vi2feat[i].strip(), norms[i]))
        self.logger.info('...')
        for i in reversed(norms_vi[:self.nb_samples]):
            self.logger.info('%-60s %.3lf' % (self.vi2feat[i].strip(), norms[i]))


class FeatureVectorInteractions(Layer):
    """"""

    def __init__(self, **kwargs):
        super(FeatureVectorInteractions, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs):
        VI0, VI1, V0, V1 = inputs

        # Multiply the feature matrices to get *b* matrices of pair-wise feature products.
        P = K.batch_dot(V0, K.permute_dimensions(V1, (0, 2, 1)), (2, 1))

        # Each of the VI0 and VI1 has a number of zero entries which should be
        # masked out. Compute row and column masks identifying non-zero entries.
        # Row mask must be permuted to represent rows instead of cols.
        row_mask = K.repeat(K.clip(VI0, 0, 1), P.shape[1])
        row_mask = K.permute_dimensions(row_mask, (0, 2, 1))
        col_mask = K.repeat(K.clip(VI1, 0, 1), P.shape[1])

        # Combine the row and col masks into combined via elem-wise multiply.
        cmb_mask = row_mask * col_mask

        # Apply the combined mask to the product matrices via elem-wise multiply.
        P = P * cmb_mask

        # Return the sum of each matrix. This leaves a single scalar for each
        # pair of feature matrices, representing their sum of interactions.
        return K.expand_dims(K.sum(P, (1, 2)))

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)


class IAFM(object):

    def __init__(self, key2feats, best_model_path, vec_size, vecs_init_func,
                 vecs_init_kwargs, vecs_reg_func, vecs_reg_kwargs, optimizer,
                 optimizer_kwargs, dropout_prop, nb_epochs_max, batch_size,
                 early_stop_metric, early_stop_delta, early_stop_patience):

        self.key2feats = key2feats
        self.best_model_path = best_model_path
        self.vec_size = vec_size
        self.vecs_init_func = vecs_init_func
        self.vecs_init_kwargs = vecs_init_kwargs
        self.vecs_reg_func = vecs_reg_func
        self.vecs_reg_kwargs = vecs_reg_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.dropout_prop = dropout_prop
        self.nb_epochs_max = nb_epochs_max
        self.batch_size = batch_size
        self.early_stop_metric = early_stop_metric
        self.early_stop_delta = early_stop_delta
        self.early_stop_patience = early_stop_patience
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info('Preprocessing features')
        # Compute all of the unique feature keys.
        # Sort them to ensure reproducibility (as long as no new features are added).
        feats_unique = set(flatten(self.key2feats.values()))
        feats_unique = sorted(feats_unique, key=str)

        # Map feature string -> vector index and vice-versa. Add 1 for padding index.
        self.feat2vi = {f: i for i, f in enumerate(['padding'] + feats_unique)}
        self.vi2feat = {i: f for f, i in self.feat2vi.items()}

        # Number of vectors required. One per feature plus padding vector.
        self.nb_vecs = len(self.feat2vi) + 1

        # Max number of feature indexes.
        self.nb_vi_max = max(map(len, self.key2feats.values()))
        self.logger.info('Vector space dimensions: (%dx%d)' % (self.nb_vecs, self.nb_vi_max))

        # Populate the VI matrix. Each row contains vector indexes for single key.
        # Also need a mapping from key -> vector indexes for that key.
        self.VI = np.zeros((self.nb_vecs, self.nb_vi_max), dtype=np.uint32)
        self.key2VI_idx = {}
        for i, (key, feats) in enumerate(self.key2feats.items()):
            self.VI[i, :len(feats)] = [self.feat2vi[f] for f in feats]
            self.key2VI_idx[key] = i

    def net(self):

        # Three inputs: user vector indexes, song vector indexes.
        VI0 = Input((self.nb_vi_max,), name='VI0')
        VI1 = Input((self.nb_vi_max,), name='VI1')

        # Setup vector space regularization
        reg = self.vecs_reg_func(**self.vecs_reg_kwargs)

        # Initialize vector space weights. Vector 0 is all 0s.
        self.vecs_init_kwargs.update({'size': (self.nb_vecs, self.vec_size)})
        W = self.vecs_init_func(**self.vecs_init_kwargs)
        W[0] *= 0

        # Build and initialize single vector space.
        V = Embedding(self.nb_vecs, self.vec_size, name='vecs', weights=[W],
                      embeddings_regularizer=reg)

        # Compute feature vector interactions. Return *batch* scalars.
        I = FeatureVectorInteractions(name='fvi')([VI0, VI1, V(VI0), V(VI1)])

        # Apply sigmoid for classification and return network.
        classify = Activation('sigmoid')(I)
        return Model([VI0, VI1], classify)

    def gen(self, X, y, shuffle, dropout_prop):
        bii = np.arange(len(X))
        while True:
            if shuffle:
                np.random.shuffle(bii)
            for bii_ in chunked(bii, self.batch_size):
                ii0 = [self.key2VI_idx[k] for k in X[bii_, 0]]
                ii1 = [self.key2VI_idx[k] for k in X[bii_, 1]]
                yield [self.VI[ii0], self.VI[ii1]], y[bii_]

    def fit(self, Xt, Xv, yt, yv):

        def auc_roc(yt, yp):
            """https://github.com/fchollet/keras/issues/6050"""
            value, update_op = tf.contrib.metrics.streaming_auc(yp, yt)
            metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]
            for v in metric_vars:
                tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)
            with tf.control_dependencies([update_op]):
                value = tf.identity(value)
                return value

        def avg_pos(yt, yp):
            return K.sum(yp * yt) / K.sum(yt)

        def avg_neg(yt, yp):
            return K.sum(yp * (1 - yt)) / K.sum(1 - yt)

        net = self.net()
        net.summary()

        opt = self.optimizer(**self.optimizer_kwargs)
        gen_trn = self.gen(Xt, yt, shuffle=True, dropout_prop=self.dropout_prop)
        gen_val = self.gen(Xv, yv, shuffle=True, dropout_prop=0.)

        cb = [
            VecReviewCallback(self.vi2feat),
            ModelCheckpoint(self.best_model_path, monitor=self.early_stop_metric, verbose=1,
                            save_best_only=True, save_weights_only=True, mode='max'),
            EarlyStopping(monitor=self.early_stop_metric, min_delta=self.early_stop_delta,
                          patience=self.early_stop_patience, verbose=1, mode='max'),
        ]

        net.compile(optimizer=opt, loss='binary_crossentropy', metrics=[auc_roc, avg_pos, avg_neg])
        history = net.fit_generator(
            epochs=self.nb_epochs_max, verbose=1, callbacks=cb, max_queue_size=1000,
            generator=gen_trn, steps_per_epoch=ceil(len(Xt) / self.batch_size),
            validation_data=gen_val, validation_steps=ceil(len(Xv) / self.batch_size)
        )

        i = np.argmax(history.history['val_auc_roc'])
        return history.history['val_loss'][i], history.history['val_auc_roc'][i]

    def predict(self, X):
        net = self.net()
        net.load_weights(self.best_model_path, by_name=True)
        gen = self.gen(X, y=np.zeros(len(X)), shuffle=False, dropout_prop=0)
        yp = []
        for _ in tqdm(range(0, len(X), self.batch_size)):
            X_, _ = next(gen)
            yp_ = net.predict(X_, batch_size=self.batch_size)
            yp += yp_[:, 0].tolist()
        return yp


def round_to(n, r):
    return round(n / r) * r


def get_user_feats(df, logger):
    """For each user, create a mapping from a unique user key to
    a list of [feature type, feature value] pairs."""

    # Key is the user id prefixed by "u-".
    msno2key = lambda x: 'u-%s' % x
    keys = df['msno'].apply(msno2key).values.tolist()

    # Keep a lookup of the training msnos.
    msno_trn = set(df.msno.values[:NTRN])

    # Remove duplicate based on Id.
    df = df.drop_duplicates('msno')
    keys_dedup = df['msno'].apply(msno2key).values.tolist()

    # Build mapping from unique keys to features.
    key2feats = {}

    for k, (i, row) in tqdm(zip(keys_dedup, df.iterrows())):

        key2feats[k] = []

        # User msno (id). Missing if not in training set.
        if row['msno'] in msno_trn:
            key2feats[k].append('u-msno-%s' % row['msno'])

        # User age. Clipped and rounded.
        if 0 < row['bd'] < 70:
            key2feats[k].append('u-age-%d' % round_to(row['bd'], 5))

        # User city. No missing values.
        key2feats[k].append('u-city-%d' % int(row['city']))

        # User gender. Missing if not female or male.
        if row['gender'] in {'male', 'female'}:
            key2feats[k].append('u-sex-%s' % row['gender'])

        # User registration method. No missing values.
        key2feats[k].append('u-reg-via-%d' % int(row['registered_via']))

        # User registration year. No missing values.
        y0 = int(str(row['registration_init_time'])[:4])
        key2feats[k].append('u-reg-year-%d' % y0)

        assert len(key2feats[k]) > 0, 'No features found for %s' % k

    return keys, key2feats


def get_song_feats(df, logger):

    # Key is the song id prefixed by "s-"
    # Need to get all of the keys first, including duplicates.
    songid2key = lambda x: 's-%s' % x
    keys = df['song_id'].apply(songid2key).values.tolist()

    # Keep only the song-related columns and remove duplicates based on song_id.
    cols = ['song_id', 'song_length', 'name', 'language', 'isrc', 'genre_ids', 'artist_name', 'composer', 'lyricist']
    hash_cols = cols[1:]
    df = df[cols].drop_duplicates('song_id')
    keys_dedup = df['song_id'].apply(songid2key).values.tolist()

    # Replace unknown artists, lyricists, composers. '佚名' means missing names.
    # https://www.kaggle.com/c/kkbox-music-recommendation-challenge/discussion/43645
    df['artist_name'].replace('佚名', np.nan, inplace=True)
    df['lyricist'].replace('佚名', np.nan, inplace=True)
    df['composer'].replace('佚名', np.nan, inplace=True)

    # Replace languages. 3,10,24,59 represent Chinese.
    # https://www.kaggle.com/c/kkbox-music-recommendation-challenge/discussion/43645
    df['language'].replace(10, 3, inplace=True)
    df['language'].replace(24, 3, inplace=True)
    df['language'].replace(59, 3, inplace=True)

    # Build mapping from unique keys to features.
    key2feats = {}

    # Regular expressions and function for parsing musicians.
    RE_MUSICIANS_SPLIT_PATTERN = re.compile(r'feat(.)\w*|\(|\)|\|')

    def parse_musicians(mstr):
        r = []
        if type(mstr) is str:
            mstr = mstr.lower().replace('\n', ' ')
            s = re.sub(RE_MUSICIANS_SPLIT_PATTERN, ',', mstr).split(',')
            for t in s:
                t = t.strip()
                if len(t) > 0:
                    r.append('s-musician-%s' % t)
        return r

    all_musicians = set()

    for k, (i, row) in tqdm(zip(keys_dedup, df.iterrows())):

        key2feats[k] = []

        # Ad-hoc replacements.
        if row['song_id'] == 'GQK/aYFW8elL+4o7Qo1zNSQmjYDKJfacYT2+QKWQ71U=':
            row['isrc'] = 'ESAAI0205357'

        # Song hash. There are ~9K song records that have distinct song_ids
        # but otherwise identical properties. Use the hash of these properties
        # instead of the song_id as a unique identifier.
        song_hash = md5(str(row[hash_cols].values).encode()).hexdigest()
        key2feats[k].append('s-hash-%s' % song_hash)

        # Song length. Missing if nan. Log transformed.
        if not np.isnan(row['song_length']):
            f = round(log(1 + row['song_length'] / 1000 / 60))
            key2feats[k].append('s-len-%d' % f)

        # Song language. Missing if nan.
        if not np.isnan(row['language']):
            key2feats[k].append('s-lang-%d' % row['language'])

        # Song year. Missing if nan. Rounded to 3-year intervals.
        # Song country. Missing if nan.
        if type(row['isrc']) is str:
            f = int(round_to(int(row['isrc'][5:7]), 3))
            key2feats[k].append('s-year-%d' % f)
            key2feats[k].append('s-country-%s' % row['isrc'][:2])

        # Song genre(s). Missing if nan. Split on pipes.
        if type(row['genre_ids']) is str:
            for x in row['genre_ids'].split('|'):
                key2feats[k].append('s-genre-%d' % int(x))

        mm = parse_musicians(row['artist_name'])
        mm += parse_musicians(row['composer'])
        mm += parse_musicians(row['lyricist'])
        key2feats[k] += list(set(mm))

        assert len(key2feats[k]) > 0, 'No features found for %s' % k

    return keys, key2feats


class MultiVecRec(object):

    def __init__(self,
                 data_dir,
                 artifacts_dir,
                 best_model_path,
                 predict_path):

        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        self.best_model_path = best_model_path
        self.predict_path = predict_path
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_features(self, train=False, test=False):

        path_sample_keys_trn = '%s/data-sample-keys-trn.csv' % self.artifacts_dir

        path_sample_keys_tst = '%s/data-sample-keys-tst.csv' % self.artifacts_dir
        path_feats = '%s/data-feats.json' % self.artifacts_dir

        pp = [path_sample_keys_trn, path_sample_keys_tst, path_feats]
        feats_ready = sum([exists(p) for p in pp]) == len(pp)

        if not feats_ready:

            self.logger.info('Reading dataframes')
            SNG = pd.read_csv('%s/songs.csv' % self.data_dir)
            SEI = pd.read_csv('%s/song_extra_info.csv' % self.data_dir)
            MMB = pd.read_csv('%s/members.csv' % self.data_dir)
            TRN = pd.read_csv('%s/train.csv' % self.data_dir)
            TST = pd.read_csv('%s/test.csv' % self.data_dir)

            self.logger.info('Merge TRN, TST with SNG, SEI, MMB.')
            TRN = TRN.merge(MMB, on='msno', how='left')
            TST = TST.merge(MMB, on='msno', how='left')
            TRN = TRN.merge(SNG, on='song_id', how='left')
            TST = TST.merge(SNG, on='song_id', how='left')
            TRN = TRN.merge(SEI, on='song_id', how='left')
            TST = TST.merge(SEI, on='song_id', how='left')

            self.logger.info('Combining TRN and TST')
            CMB = TRN.append(TST)

            # Throw away unused after merging.
            del SNG, SEI, MMB

            self.logger.info('Encoding user features')
            ukeys, ukey2feats = get_user_feats(CMB, self.logger)

            self.logger.info('Encoding song features')
            skeys, skey2feats = get_song_feats(CMB, self.logger)

            self.logger.info('Saving features')
            with open(path_feats, 'w') as fp:
                feats = ukey2feats.copy()
                feats.update(skey2feats)
                json.dump(feats, fp, sort_keys=True, indent=2)

            self.logger.info('Saving training keys')
            keys_trn = pd.DataFrame({
                'user': ukeys[:min(NTRN, len(TRN))],
                'song': skeys[:min(NTRN, len(TRN))],
                'target': TRN['target']
            })
            keys_trn.to_csv(path_sample_keys_trn, index=False)

            self.logger.info('Saving testing keys')
            keys_tst = pd.DataFrame({
                'id': TST['id'],
                'user': ukeys[-min(NTST, len(TST)):],
                'song': skeys[-min(NTST, len(TST)):]
            })
            keys_tst.to_csv(path_sample_keys_tst, index=False)

            # How many warm-start users?
            U = set(keys_trn.user)
            n = np.sum(keys_tst.user.apply(lambda x: x in U))
            self.logger.info('Warm start users: %d, %.3lf' % (n, n / len(keys_tst)))

            # How many warm-start songs?
            S = set(keys_trn.song)
            n = np.sum(keys_tst.song.apply(lambda x: x in S))
            self.logger.info('Warm start songs: %d, %.3lf' % (n, n / len(keys_tst)))

        # Read from disk and return keys and features.
        self.logger.info('Reading features from disk')
        keys = pd.read_csv(path_sample_keys_trn) if train \
            else pd.read_csv(path_sample_keys_tst)
        with open(path_feats, 'r') as fp:
            feats = json.load(fp)

        return keys, feats

    def fit(self, samples, key2feats, IAFM_kwargs=IAFM_HYPERPARAMS_DEFAULT):
        iafm = IAFM(key2feats, self.best_model_path, **IAFM_kwargs)
        X = samples[['user', 'song']].values
        y = samples['target'].values
        _, Xv, _, yv = train_test_split(X, y, test_size=0.05, shuffle=False)
        val_loss, val_auc = iafm.fit(X, Xv, y, yv)

    def val(self, samples, key2feats, IAFM_kwargs=IAFM_HYPERPARAMS_DEFAULT):
        """
        Cold-start users and songs in test set:
        tst users in trn: 0.928
        tst songs in trn: 0.875

        Using last 40% of rows for validation:
        val users in trn: 0.910
        val songs in trn: 0.873
        """
        self.logger.info(pformat(IAFM_kwargs))
        iafm = IAFM(key2feats, self.best_model_path, **IAFM_kwargs)

        # Split features.
        X = samples[['user', 'song']].values
        y = samples['target'].values
        Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.5, shuffle=False)

        # Display cold-start proportions.
        s = set(Xt[:, 0])
        n = sum(map(lambda x: x not in s, Xv[:, 0]))
        self.logger.info('Cold-start users = %d, %.2lf' % (n, n / len(Xv)))

        s = set(Xt[:, 1])
        n = sum(map(lambda x: x not in s, Xv[:, 1]))
        self.logger.info('Cold-start songs = %d, %.2lf' % (n, n / len(Xv)))

        # Train.
        val_loss, val_auc = iafm.fit(Xt, Xv, yt, yv)

    def predict(self, samples, key2feats, IAFM_kwargs=IAFM_HYPERPARAMS_DEFAULT):
        iafm = IAFM(key2feats, self.best_model_path, **IAFM_kwargs)
        yp = iafm.predict(samples[['user', 'song']].values)
        self.logger.info('yp mean=%.3lf' % (np.mean(yp)))
        df = pd.DataFrame({'id': samples['id'], 'target': yp})
        df.to_csv(self.predict_path, index=False)
        self.logger.info('Saved %s' % self.predict_path)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description='')
    ap.add_argument('--fit', action='store_true', default=False)
    ap.add_argument('--val', action='store_true', default=False)
    ap.add_argument('--predict', action='store_true', default=False)
    args = vars(ap.parse_args())

    model = MultiVecRec(
        data_dir='data',
        artifacts_dir='artifacts/multivecrec',
        best_model_path='artifacts/multivecrec/model-iafm-best.hdf5',
        predict_path='artifacts/multivecrec/predict_tst_%d.csv' % int(time())
    )

    if args['fit']:
        model.fit(*model.get_features(train=True))

    if args['predict']:
        model.predict(*model.get_features(test=True))

    if args['val']:
        model.val(*model.get_features(train=True))
