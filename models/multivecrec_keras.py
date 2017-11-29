from collections import Counter
from math import log, ceil
from more_itertools import flatten, chunked
from multiprocessing import Pool, cpu_count
from os.path import exists
from os import getenv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, ShuffleSplit
from time import time
from tqdm import tqdm
import argparse
import logging
import numpy as np
import pandas as pd
import pickle
import pdb
import tensorflow as tf

NTRN = 7377418
NTST = 2556790
np.random.seed(865)

assert getenv('CUDA_VISIBLE_DEVICES') is not None, "Specify a GPU"
assert len(getenv('CUDA_VISIBLE_DEVICES')) > 0, "Specify a GPU"

from keras.layers import Input, Embedding, Activation, Lambda, concatenate, multiply
from keras.models import Model, load_model
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend as K

# 7 epochs.
IAFM_HYPERPARAMS_DEFAULT = {
    'vec_size': 25,
    'vecs_init_func': np.random.normal,
    'vecs_init_kwargs': {'loc': 0, 'scale': 0.01},
    'vecs_reg_func': l2,
    'vecs_reg_kwargs': {'l': 0.0},
    'optimizer': Adam,
    'optimizer_kwargs': {'lr': 0.01},
    'nb_epochs_max': 3,
    'batch_size': 50000,
    'early_stop_metric': 'val_auc_roc',
    'early_stop_delta': 0.005,
    'early_stop_patience': 2,
}


class IAFM(object):

    def __init__(self, key2feats, best_model_path, vec_size, vecs_init_func,
                 vecs_init_kwargs, vecs_reg_func, vecs_reg_kwargs, optimizer,
                 optimizer_kwargs, nb_epochs_max, batch_size, early_stop_metric,
                 early_stop_delta, early_stop_patience,):

        self.key2feats = key2feats
        self.best_model_path = best_model_path
        self.vec_size = vec_size
        self.vecs_init_func = vecs_init_func
        self.vecs_init_kwargs = vecs_init_kwargs
        self.vecs_reg_func = vecs_reg_func
        self.vecs_reg_kwargs = vecs_reg_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.nb_epochs_max = nb_epochs_max
        self.batch_size = batch_size
        self.early_stop_metric = early_stop_metric
        self.early_stop_delta = early_stop_delta
        self.early_stop_patience = early_stop_patience

        # Compute all of the unique feature keys.
        feats_unique = set(flatten(self.key2feats.values()))
        feats_unique = sorted(feats_unique, key=str)

        # Map feature -> vector index. Add 1 to account for padding index.
        self.feat2vii = {f: i + 1 for i, f in enumerate(feats_unique)}

        # Number of vectors required. One per feature plus padding vector.
        self.nb_vecs = max(self.feat2vii.values()) + 1

        # Max number of feature indexes.
        self.nb_vii_max = max(map(len, self.key2feats.values()))

        # Map key -> [padded list of the indexes]. 0 used for padding.
        # Map key -> number of non-padded indexes in index list
        self.key2vii = {}
        self.key2len = {}
        for k, v in self.key2feats.items():
            padding = [0] * (self.nb_vii_max - len(v))
            self.key2vii[k] = [self.feat2vii[f] for f in v] + padding
            self.key2len[k] = len(v)
        assert set(map(len, self.key2vii.values())) == {self.nb_vii_max}

    def net(self):

        # Three inputs: vector indexes, vector indexes, interaction coefficients.
        feat_vii_0 = Input((self.nb_vii_max,), name='feat_vii_0')
        feat_vii_1 = Input((self.nb_vii_max,), name='feat_vii_1')
        inter_coefs = Input((1,), name='inter_coefs')

        # Build and initialize single vector space. Set vector 0 to all 0s.
        self.vecs_init_kwargs.update({'size': (self.nb_vecs, self.vec_size)})
        vecsw = self.vecs_init_func(**self.vecs_init_kwargs)
        vecsw[0] *= 0
        reg = self.vecs_reg_func(**self.vecs_reg_kwargs)
        vecs = Embedding(self.nb_vecs, self.vec_size, weights=[vecsw],
                         mask_zero=True, name='vecs', embeddings_regularizer=reg)

        # Dot and adjust vectors.
        vecs_cat = concatenate([vecs(feat_vii_0), vecs(feat_vii_1)], axis=-1, name='vecs_cat')
        b = lambda v0, v1: K.batch_dot(v0, v1, axes=(2, 1))
        t = lambda v: K.permute_dimensions(v, (0, 2, 1))
        vecs_mul = Lambda(lambda x: b(x[:, :, :self.vec_size], t(x[:, :, self.vec_size:])), name='vecs_mul')(vecs_cat)
        vecs_sum = Lambda(lambda x: K.sum(x, (1, 2)), name='vecs_sum')(vecs_mul)
        vecs_adj = multiply([vecs_sum, inter_coefs], name='vecs_adj')
        classify = Activation('sigmoid')(vecs_adj)

        net = Model([feat_vii_0, feat_vii_1, inter_coefs], classify)

        # Ensure padding vector is all zeros.
        assert set(net.get_layer('vecs').get_weights()[0][0]) == {0}

        return net

    def gen(self, X, y, shuffle):
        bii = np.arange(len(X))
        while True:
            if shuffle:
                np.random.shuffle(bii)
            for bii_ in chunked(bii, self.batch_size):
                X_0, X_1, X_2 = [], [], []
                for i, (k0, k1) in enumerate(X[bii_]):
                    X_0.append(self.key2vii[k0])
                    X_1.append(self.key2vii[k1])

                    if self.key2len[k0] * self.key2len[k1] == 0:
                        pdb.set_trace()

                    X_2.append(1. / (self.key2len[k0] * self.key2len[k1]))
                X_0 = np.array(X_0)
                X_1 = np.array(X_1)
                X_2 = np.array(X_2)
                yield [X_0, X_1, X_2], y[bii_]

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

        net = self.net()
        opt = self.optimizer(**self.optimizer_kwargs)
        gen_trn = self.gen(Xt, yt, shuffle=True)
        gen_val = self.gen(Xv, yv, shuffle=True)

        cb = [
            ModelCheckpoint(self.best_model_path, monitor=self.early_stop_metric, verbose=1,
                            save_best_only=True, save_weights_only=True, mode='max'),
            EarlyStopping(monitor=self.early_stop_metric, min_delta=self.early_stop_delta,
                          patience=self.early_stop_patience, verbose=1, mode='max')
        ]

        net.compile(optimizer=opt, loss='binary_crossentropy', metrics=[auc_roc])
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
        gen = self.gen(X, y=np.zeros(len(X)), shuffle=False)
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
            key2feats[k].append(('u-msno', row['msno']))

        # User age. Clipped and rounded.
        if 0 < row['bd'] < 70:
            key2feats[k].append(('u-age', round_to(row['bd'], 5)))

        # User city. No missing values.
        key2feats[k].append(('u-city', int(row['city'])))

        # User gender. Missing if not female or male.
        if row['gender'] in {'male', 'female'}:
            key2feats[k].append(('u-sex', row['gender']))

        # User registration method. No missing values.
        key2feats[k].append(('u-reg-via', int(row['registered_via'])))

        # User registration year. No missing values.
        y0 = int(str(row['registration_init_time'])[:4])
        key2feats[k].append(('u-reg-year', y0))

        assert len(key2feats[k]) > 0, 'No features found for %s' % k

    return keys, key2feats


def split_multi(maybe_vals):
    if type(maybe_vals) == str:
        split = maybe_vals.split('|')
        try:
            return [int(x) for x in split]
        except ValueError as ex:
            return [x.strip() for x in split]
    return []


def get_song_feats(df, logger):

    # Key is the song id prefixed by "s-"
    songid2key = lambda x: 's-%s' % x
    keys = df['song_id'].apply(songid2key).values.tolist()

    # Count the instances of each musician and genre.
    # Musicians encompass artist, composer, and lyricist.
    t0 = time()
    pool = Pool(cpu_count())
    musicians = flatten(pool.map(split_multi,
                                 df['artist_name'].values.tolist() +
                                 df['lyricist'].values.tolist() +
                                 df['composer'].values.tolist()))
    musician_counts = Counter(musicians)
    logger.info('Counted musicians in %d seconds' % (time() - t0))

    genre_ids = flatten(pool.map(split_multi, df['genre_ids'].values.tolist()))
    genre_id_counts = Counter(genre_ids)
    logger.info('Counted genres in %d seconds' % (time() - t0))
    pool.close()

    # Keep a lookup of the training song ids.
    song_ids_trn = set(df['song_id'].values[:NTRN])

    # Remove duplicates.
    df = df.drop_duplicates('song_id')
    keys_dedup = df['song_id'].apply(songid2key).values.tolist()

    # Build mapping from unique keys to features.
    key2feats = {}

    for k, (i, row) in tqdm(zip(keys_dedup, df.iterrows())):

        key2feats[k] = []

        # Ad-hoc replacements.
        if row['song_id'] == 'GQK/aYFW8elL+4o7Qo1zNSQmjYDKJfacYT2+QKWQ71U=':
            row['isrc'] = 'ESAAI0205357'

        # Song id. Missing if not in training set.
        if row['song_id'] in song_ids_trn:
            key2feats[k].append(('s-id', row['song_id']))

        # Song length. Missing if nan.
        if not np.isnan(row['song_length']):
            f = row['song_length'] / 1000 / 60
            f = round(log(max(1, f)))
            key2feats[k].append(('s-len', f))

        # Song language. Missing if nan.
        if not np.isnan(row['language']):
            key2feats[k].append(('s-lang', int(row['language'])))

        # Song year. Missing if nan. Rounded to 3-year intervals.
        # Song country. Missing if nan.
        if type(row['isrc']) is str:
            f = int(round_to(int(row['isrc'][5:7]), 3))
            key2feats[k].append(('s-year', f))
            key2feats[k].append(('s-country', row['isrc'][:2]))

        # Song genre(s). Missing if nan. Split on pipes.
        gg = split_multi(row['genre_ids'])
        if len(gg) > 0:
            ggc = map(genre_id_counts.get, gg)
            key2feats[k].append(('s-genre', gg[np.argmax(ggc)]))

        # TODO: consider artist, lyricist, musician separately.
        mm = split_multi(row['artist_name'])
        mm += split_multi(row['lyricist'])
        mm += split_multi(row['composer'])
        if len(mm) > 0:
            mmc = map(musician_counts.get, mm)
            key2feats[k].append(('s-musician', mm[np.argmax(mmc)]))

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
        path_feats = '%s/data-feats.pkl' % self.artifacts_dir

        pp = [path_sample_keys_trn, path_sample_keys_tst, path_feats]
        feats_ready = sum([exists(p) for p in pp]) == len(pp)

        if not feats_ready:

            self.logger.info('Reading dataframes')
            SNG = pd.read_csv('%s/songs.csv' % self.data_dir)
            SEI = pd.read_csv('%s/song_extra_info.csv' % self.data_dir, usecols=['song_id', 'isrc'])
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
            with open(path_feats, 'wb') as fp:
                feats = ukey2feats.copy()
                feats.update(skey2feats)
                pickle.dump(feats, fp)

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

            # Count occurrences of test features in training set.
            with open(path_feats, 'rb') as fp:
                feats = pickle.load(fp)
            keys_trn = pd.read_csv(path_sample_keys_trn, usecols=['user', 'song'])
            keys_tst = pd.read_csv(path_sample_keys_tst, usecols=['user', 'song'])
            feats_trn = flatten(map(feats.get, flatten(keys_trn.values)))
            feats_tst = flatten(map(feats.get, flatten(keys_tst.values)))
            cntr_trn = Counter(feats_trn)
            cntr_tst = Counter({k: cntr_trn[k] for k in set(feats_tst)})
            cnts_trn = np.array(list(cntr_trn.values()))
            cnts_tst = np.array(list(cntr_tst.values()))

            self.logger.info('Training set most common features')
            self.logger.info(cntr_trn.most_common()[:6])
            self.logger.info('Test set most common features')
            self.logger.info(cntr_tst.most_common()[:6])
            self.logger.info('Training set feature counts:')
            self.logger.info('mean = %.2lf' % cnts_trn.mean())
            self.logger.info('median = %d' % np.median(cnts_trn))
            self.logger.info('max = %d' % cnts_trn.max())
            self.logger.info('min = %d' % cnts_trn.min())
            self.logger.info('Test set feature counts (occurring in the training set):')
            self.logger.info('mean = %.2lf' % cnts_tst.mean())
            self.logger.info('median = %d' % np.median(cnts_tst))
            self.logger.info('max = %d' % cnts_tst.max())
            self.logger.info('min = %d' % cnts_tst.min())

        # Read from disk and return keys and features.
        self.logger.info('Reading features from disk')
        keys = pd.read_csv(path_sample_keys_trn) if train \
            else pd.read_csv(path_sample_keys_tst)
        with open(path_feats, 'rb') as fp:
            feats = pickle.load(fp)

        return keys, feats

    def fit(self, samples, key2feats, IAFM_kwargs=IAFM_HYPERPARAMS_DEFAULT):
        iafm = IAFM(key2feats, self.best_model_path, **IAFM_kwargs)
        X = samples[['user', 'song']].values
        y = samples['target'].values
        _, Xv, _, yv = train_test_split(X, y, test_size=0.2)
        val_loss, val_auc = iafm.fit(X, Xv, y, yv)

    def crossval(self, samples, key2feats, IAFM_kwargs=IAFM_HYPERPARAMS_DEFAULT):
        X = samples[['user', 'song']].values
        y = samples['target'].values
        losses, aucs = [], []
        splitter = ShuffleSplit(n_splits=4, test_size=0.25)
        s = 'Starting %d-fold cross-validation' % splitter.n_splits
        self.logger.info('-' * len(s))
        self.logger.info(s)
        self.logger.info('-' * len(s))
        for cvi, (ii_trn, ii_val) in enumerate(splitter.split(X, y)):
            iafm = IAFM(key2feats, self.best_model_path, **IAFM_kwargs)
            val_loss, val_auc = iafm.fit(X[ii_trn], X[ii_val], y[ii_trn], y[ii_val])
            losses.append(val_loss)
            aucs.append(val_auc)
            s = '%d: mean loss = %.3lf, mean auc = %.3lf' % (cvi, np.mean(losses), np.mean(aucs))
            self.logger.info('-' * len(s))
            self.logger.info(s)
            self.logger.info('-' * len(s))
        return np.mean(losses), np.mean(aucs)

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
    ap.add_argument('--predict', action='store_true', default=False)
    ap.add_argument('--crossval', action='store_true', default=False)
    args = vars(ap.parse_args())

    model = MultiVecRec(
        data_dir='data',
        artifacts_dir='artifacts/multivecrec',
        best_model_path='artifacts/multivecrec/model-iafm-best.hdf5',
        predict_path='artifacts/multivecrec/predict_tst_%d.csv' % int(time())
    )

    # model.get_features()

    if args['fit']:
        model.fit(*model.get_features(train=True))

    if args['predict']:
        model.predict(*model.get_features(test=True))

    if args['crossval']:
        model.crossval(*model.get_features(train=True))
