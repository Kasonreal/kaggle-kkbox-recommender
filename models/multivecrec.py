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

NTRN = 7377418
NTST = 2556790
np.random.seed(865)

assert getenv('CUDA_VISIBLE_DEVICES') is not None, "Specify a GPU"
assert len(getenv('CUDA_VISIBLE_DEVICES')) > 0, "Specify a GPU"

from keras.engine.topology import Layer
from keras.layers import Input, Embedding, Activation, Lambda, concatenate, multiply, dot
from keras.models import Model, load_model
from keras.initializers import RandomNormal
from keras.optimizers import Adam, SGD, Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, CSVLogger
import keras.backend as K


IAFM_HYPERPARAMS_DEFAULT = {
    'vec_size': 60,
    'vecs_init_func': np.random.normal,
    'vecs_init_kwargs': {'loc': 0, 'scale': 0.1},
    'dropout_prop': 0.55,
    'optimizer': Adagrad,
    'optimizer_kwargs': {'lr': 0.01},
    'nb_epochs_max': 15,
    'batch_size': 50000,
    'early_stop_metric': 'val_auc',
    'early_stop_delta': 0.005,
    'early_stop_patience': 10,
}


class AUCCallback(Callback):

    def __init__(self, Xt, Xv, yt, yv, batch_size):
        self.Xt = Xt
        self.Xv = Xv
        self.yt = yt
        self.yv = yv
        self.batch_size = batch_size
        self.logger = logging.getLogger(self.__class__.__name__)

    def on_epoch_end(self, batch, logs):
        yp = self.model.predict(self.Xt, batch_size=self.batch_size)
        logs['auc'] = roc_auc_score(self.yt, yp)
        self.logger.info('Trn AUC = %.4lf' % logs['auc'])
        if len(self.yt) == len(self.yv) and np.all(self.yt == self.yv):
            logs['val_auc'] = logs['auc']
        else:
            yp = self.model.predict(self.Xv, batch_size=self.batch_size)
            logs['val_auc'] = roc_auc_score(self.yv, yp)
        self.logger.info('Val AUC = %.4lf' % logs['val_auc'])


class VecReviewCallback(Callback):

    def __init__(self, vi2feat, feat2cnt, nb_samples=4):
        self.vi2feat = vi2feat
        self.feat2cnt = feat2cnt
        self.nb_samples = nb_samples
        self.logger = logging.getLogger(self.__class__.__name__)

    def on_epoch_end(self, epoch, logs):
        print('\n')

        # Extract the vectors.
        vecs = self.model.get_layer('vecs').get_weights()[0]

        # Make sure the padding vector has not been changed.
        assert np.all(vecs[0] == vecs[0] * 0), 'Vector 0 must be all-zero'

        # Compute and sort vector norms. Print those with highest and lowest norm.
        norms = np.sqrt(np.sum(vecs ** 2, 1))
        norms_vi = np.argsort(norms)
        for i in reversed(norms_vi[-self.nb_samples:]):
            f = self.vi2feat[i]
            self.logger.info('%-60s %d %.3lf' % (f, self.feat2cnt[f], norms[i]))
        self.logger.info('...')
        for i in reversed(norms_vi[:self.nb_samples]):
            f = self.vi2feat[i]
            self.logger.info('%-60s %d %.3lf' % (f, self.feat2cnt[f], norms[i]))


class FeatureVectorInteractions(Layer):
    """"""

    def __init__(self, dropout_prop, **kwargs):
        super(FeatureVectorInteractions, self).__init__(**kwargs)
        self.uses_learning_phase = True  # Needed for dropout.
        self.dropout_prop = dropout_prop

    def build(self, input_shape):
        pass

    def call(self, inputs):

        VI0 = inputs[0]  # Vector indexes representing each user's properties.
        VI1 = inputs[1]  # Vector indexes representing each item's properties.
        V0 = inputs[2]   # Matrix of row vectors for each user's properties.
        V1 = inputs[3]   # Matrix of row vectors for each item's properties.

        # Multiply the feature matrices to get *b* matrices of pair-wise feature products.
        P = K.batch_dot(V0, K.permute_dimensions(V1, (0, 2, 1)), (2, 1))

        # Each of the VI0 and VI1 has a number of zero entries which should be
        # masked out. Compute row and column masks identifying non-zero entries.
        # Row mask must be permuted to represent rows instead of cols.
        row_masks = K.repeat(K.clip(VI0, 0, 1), P.shape[1])
        row_masks = K.permute_dimensions(row_masks, (0, 2, 1))
        col_masks = K.repeat(K.clip(VI1, 0, 1), P.shape[1])

        # Combine the row and col masks into masks where active (non-padded)
        # elements have value 1 and padding elements have value 0. This is
        # a unique mask for each product matrix.
        active_masks = row_masks * col_masks

        # Apply the active mask to the product matrices via elem-wise multiply.
        P = P * active_masks

        # For dropout, compute a binary binomial mask to 0 out elements.
        if 0. < self.dropout_prop < 1.:
            P = K.switch(K.learning_phase(),
                         P * K.random_binomial(K.shape(P), 1 - self.dropout_prop),
                         P)

        # Return the sum of each product matrix, a single scalar for each
        # pair of feature matrices, representing their sum of interactions.
        return K.expand_dims(K.sum(P, (1, 2)))

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)


class IAFM(object):

    def __init__(self, key2feats, best_model_path, history_path, vec_size,
                 vecs_init_func, vecs_init_kwargs, optimizer, optimizer_kwargs,
                 dropout_prop, nb_epochs_max, batch_size, early_stop_metric,
                 early_stop_delta, early_stop_patience):

        self.key2feats = key2feats
        self.best_model_path = best_model_path
        self.history_path = history_path
        self.vec_size = vec_size
        self.vecs_init_func = vecs_init_func
        self.vecs_init_kwargs = vecs_init_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.dropout_prop = dropout_prop
        self.nb_epochs_max = nb_epochs_max

        self.batch_size = batch_size
        self.early_stop_metric = early_stop_metric
        self.early_stop_delta = early_stop_delta
        self.early_stop_patience = early_stop_patience
        self.logger = logging.getLogger(self.__class__.__name__)

        # NOTE: If you just add an arbitrary feature to EVERY item, it ends up
        # with one of the highest norms.
        # for k in self.key2feats.keys():
        #     self.key2feats[k].append('testing-norms')

        self.logger.info('Preprocessing features')
        # Compute all of the unique feature keys.
        # Sort them to ensure reproducibility (as long as no new features are added).
        feats_all = list(flatten(self.key2feats.values()))
        feats_unique = sorted(set(feats_all), key=str)

        # Map feature string -> vector index and vice-versa. Add 1 for padding index.
        self.feat2vi = {f: i for i, f in enumerate(['padding'] + feats_unique)}
        self.vi2feat = {i: f for f, i in self.feat2vi.items()}

        # with open('vi2feat.json', 'w') as fp:
        #     json.dump(self.vi2feat, fp)
        # with open('feat2vi.json', 'w') as fp:
        #     json.dump(self.feat2vi, fp)
        # pdb.set_trace()

        # Count the features.
        self.feat2cnt = Counter(feats_all)

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

    def net(self, VI_warm=set()):

        # Three inputs: user vector indexes, song vector indexes.
        VI0 = Input((self.nb_vi_max,), name='VI0')
        VI1 = Input((self.nb_vi_max,), name='VI1')

        # Initialize vector space weights.
        self.vecs_init_kwargs.update({'size': (self.nb_vecs, self.vec_size)})
        W = self.vecs_init_func(**self.vecs_init_kwargs)

        # Zero-out any cold-start vectors.
        VI_warm -= {0}
        for vi in self.vi2feat.keys():
            W[vi] *= vi in VI_warm
        self.logger.info('Nullified %d cold-start vectors' % (self.nb_vecs - len(VI_warm)))
        self.logger.info('Preserved %d warm-start vectors' % len(VI_warm))

        # NOTE: If you use embedding regularizer, even vectors which were never
        # used will be updated due to the regularization penalty. It's possible
        # that this is the cause for misleading validation scores. See discussion:
        # https://www.reddit.com/r/MachineLearning/comments/3y41si/
        # V = Embedding(self.nb_vecs, self.vec_size, name='vecs', weights=[W])
        V = Embedding(self.nb_vecs, self.vec_size, name='vecs', weights=[W])

        # Compute feature vector interactions. Return *batch* scalars.
        I = FeatureVectorInteractions(dropout_prop=self.dropout_prop, name='fvi')\
            ([VI0, VI1, V(VI0), V(VI1)])

        # Apply sigmoid for classification and return network.
        classify = Activation('sigmoid')(I)
        return Model([VI0, VI1], classify)

    def _keys_to_vector_indexes(self, X_keys):
        ii0, ii1 = [], []
        for k0, k1 in X_keys:
            ii0.append(self.key2VI_idx[k0])
            ii1.append(self.key2VI_idx[k1])
        return [self.VI[ii0], self.VI[ii1]]

    def fit(self, Xt_keys, Xv_keys, yt, yv):

        # Custom metrics.
        def pos_avg(yt, yp):
            return K.sum(yp * yt) / K.sum(yt)

        def pos_ext(yt, yp):
            return K.max(yp * yt)

        def neg_avg(yt, yp):
            return K.sum(yp * (1 - yt)) / K.sum(1 - yt)

        def neg_ext(yt, yp):
            return K.min(yp * (1 - yt))

        # Convert keys to vector indexes.
        self.logger.info('Converting keys to vector indexes')
        Xt = self._keys_to_vector_indexes(Xt_keys)
        Xv = self._keys_to_vector_indexes(Xv_keys)

        # Build network.
        VI_warm = set(Xt[0].ravel().tolist() + Xt[1].ravel().tolist())
        net = self.net(VI_warm)
        net.summary()

        # Instantiate callbacks and compile network.
        cb = [
            VecReviewCallback(self.vi2feat, self.feat2cnt),
            AUCCallback(Xt, Xv, yt, yv, self.batch_size),
            CSVLogger(self.history_path),
            ModelCheckpoint(self.best_model_path, monitor=self.early_stop_metric, verbose=1,
                            save_best_only=True, save_weights_only=True, mode='max'),
            EarlyStopping(monitor=self.early_stop_metric, min_delta=self.early_stop_delta,
                          patience=self.early_stop_patience, verbose=1, mode='max'),
        ]
        opt = self.optimizer(**self.optimizer_kwargs)
        net.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', pos_avg, neg_avg])
        history = net.fit(Xt, yt, validation_data=(Xv, yv), batch_size=self.batch_size,
                          epochs=self.nb_epochs_max, verbose=1, callbacks=cb)

        i = np.argmax(history.history['val_auc'])
        return history.history['val_loss'][i], history.history['val_auc'][i]

    def predict(self, X_keys):
        net = self.net()
        net.load_weights(self.best_model_path, by_name=True)
        X = self._keys_to_vector_indexes(X_keys)
        yp = net.predict(X, batch_size=self.batch_size, verbose=True)
        return yp[:, 0]


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
    cols = ['msno', 'bd', 'city', 'gender', 'registration_init_time']
    df = df[cols].drop_duplicates('msno')
    keys_dedup = df['msno'].apply(msno2key).values.tolist()

    # Build mapping from unique keys to features.
    key2feats = {}

    for k, (i, row) in tqdm(zip(keys_dedup, df.iterrows())):

        key2feats[k] = []

        # User msno (id). Missing if not in training set.
        if row['msno'] in msno_trn:
            key2feats[k].append('u-id::%s' % row['msno'])

        # User age. Clipped and rounded.
        if 0 < row['bd'] < 70:
            key2feats[k].append('u-age::%d' % round_to(row['bd'], 5))

        # User city. No missing values.
        key2feats[k].append('u-city::%d' % int(row['city']))

        # User gender. Missing if not female or male.
        if row['gender'] in {'male', 'female'}:
            key2feats[k].append('u-sex::%s' % row['gender'])

        # User registration year. No missing values.
        y0 = int(str(row['registration_init_time'])[:4])
        key2feats[k].append('u-reg-year::%s' % y0)

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
                    r.append(str.strip('s-musician::%s' % t))
        return r

    all_musicians = set()

    for k, (i, row) in tqdm(zip(keys_dedup, df.iterrows())):

        key2feats[k] = []

        # Song hash. There are ~9K song records that have distinct song_ids
        # but otherwise identical properties. Use the hash of these properties
        # instead of the song_id as a unique identifier.
        song_hash = md5(str(row[hash_cols].values).encode()).hexdigest()
        key2feats[k].append('s-hash::%s' % song_hash)

        # Song length. Missing if nan. Log transform and round.
        if not np.isnan(row['song_length']):
            key2feats[k].append('s-len::%d' % round(log(1 + row['song_length'])))

        # Song language. Missing if nan.
        if not np.isnan(row['language']):
            key2feats[k].append('s-lang::%d' % row['language'])

        # Song year. Missing if nan. Rounded to 3-year intervals.
        # Song country. Missing if nan.
        if type(row['isrc']) is str:
            f = int(round_to(int(row['isrc'][5:7]), 3))
            key2feats[k].append('s-year::%d' % f)
            key2feats[k].append('s-country::%s' % row['isrc'][:2])

        # Song genre(s). Missing if nan. Split on pipes.
        if type(row['genre_ids']) is str:
            for x in row['genre_ids'].split('|'):
                key2feats[k].append('s-genre::%d' % int(x))

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
                 history_path,
                 predict_path):

        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        self.history_path = history_path
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
        self.logger.info(str(IAFM_kwargs))
        iafm = IAFM(key2feats, self.best_model_path, self.history_path, **IAFM_kwargs)
        X = samples[['user', 'song']].values
        y = samples['target'].values
        val_loss, val_auc = iafm.fit(X, X, y, y)
        self.logger.info('Best val_loss=%.4lf, val_auc=%.4lf' % (val_loss, val_auc))
        self.logger.info(str(IAFM_kwargs))

    def val(self, samples, key2feats, IAFM_kwargs=IAFM_HYPERPARAMS_DEFAULT):
        """
        Cold-start users and songs in test set:
        Cold-start users: 184018, 0.072
        Cold-start songs: 320125, 0.125

        Using last 33% of rows for validation:
        Cold-start users: 202162, 0.083
        Cold-start songs: 308135, 0.127

        Using last 20% of rows for validation:
        Cold-start users = 104479, 0.071
        Cold-start songs = 126614, 0.086
        """
        self.logger.info(str(IAFM_kwargs))
        iafm = IAFM(key2feats, self.best_model_path, self.history_path, **IAFM_kwargs)

        # Split features.
        X = samples[['user', 'song']].values
        y = samples['target'].values
        Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Display cold-start proportions.
        s = set(Xt[:, 0])
        n = sum(map(lambda x: x not in s, Xv[:, 0]))
        self.logger.info('Cold-start users = %d, %.3lf' % (n, n / len(Xv)))
        s = set(Xt[:, 1])
        n = sum(map(lambda x: x not in s, Xv[:, 1]))
        self.logger.info('Cold-start songs = %d, %.3lf' % (n, n / len(Xv)))

        # Train.
        val_loss, val_auc = iafm.fit(Xt, Xv, yt, yv)

        self.logger.info('Best val_loss=%.4lf, val_auc=%.4lf' % (val_loss, val_auc))
        self.logger.info(str(IAFM_kwargs))

    def predict(self, samples, key2feats, IAFM_kwargs=IAFM_HYPERPARAMS_DEFAULT):
        iafm = IAFM(key2feats, self.best_model_path, self.history_path, **IAFM_kwargs)
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
        history_path='artifacts/multivecrec/model-iafm-history.csv',
        predict_path='artifacts/multivecrec/predict_tst_%d.csv' % int(time())
    )

    if args['fit']:
        model.fit(*model.get_features(train=True))

    if args['predict']:
        model.predict(*model.get_features(test=True))

    if args['val']:
        model.val(*model.get_features(train=True))
