from collections import Counter
from hashlib import md5
from itertools import product
from more_itertools import flatten
from pprint import pformat
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.model_selection import train_test_split
from time import time
from tqdm import tqdm
import argparse
import json
import logging
import numpy as np
import os
import pandas as pd
import pdb
import re

import lightgbm as lgbm

from keras.engine.topology import Layer
from keras.layers import Input, Embedding, Activation
from keras.models import Model, load_model
from keras.optimizers import Adagrad
from keras import backend as K

GBDT_PARAMS_DEFAULT = {

    # Defining the task.
    'objective': 'binary',
    'boosting': 'gbdt',
    'metric': 'auc',
    'train_metric': True,

    # How many learners to fit, and how long to continue without
    # improvement on the validation set.
    'num_iterations': 450,
    'early_stopping_rounds': 50,

    # TODO: explain these parameters.
    'learning_rate': 0.3,
    'max_bin': 255,

    # Constraints on the tree characteristics.
    # Generally larger values will fit better but may over-fit.
    'max_depth': 10,
    'num_leaves': 108,

    # Randomly select *bagging_fraction* of the data to fit a learner.
    # Perform bagging at every *bagging_freq* iterations.
    # Seed the random bagging with *bagging_seed*.
    'bagging_fraction': 0.95,
    'bagging_freq': 1,
    'bagging_seed': 1,

    # Randomly select *feature_fraction* of the features to fit a learner.
    # Seed the random selection with *feature_fraction_seed*.
    'feature_fraction': 0.9,
    'feature_fraction_seed': 1,

}

INTERACTION_SPACE_PARAMS_DEFAULT = {
    'nb_vecs': -1,
    'nb_dims': 50,
    'vecs_init_func': np.random.normal,
    'vecs_init_kwargs': {'loc': 0, 'scale': 0.05},
    'optimizer': Adagrad,
    'optimizer_kwargs': {'lr': 0.05},
    'nb_epochs_max': 10,
    'batch_size': 50000
}


def save_best_model(check_every_iterations, name, metric, mode, model_path, params_path=None):
    """LGBM callback."""
    metric_vals = []
    improved_iterations = [-1]
    cmpfunc = np.argmax if mode == 'max' else np.argmin

    def callback(env):
        for res in env.evaluation_result_list:
            if res[0] == name and res[1] == metric:
                metric_vals.append(res[2])
        if env.iteration % check_every_iterations != 0:
            return
        check = cmpfunc(metric_vals)
        if check > improved_iterations[-1]:
            improved_iterations.append(check)
            i = improved_iterations[-1]
            p = model_path.format(name=name, metric=metric, val=metric_vals[i])
            print('[%d] metric %s:%s improved: %.5lf on iteration %d. Saving model %s' %
                  (env.iteration, name, metric, metric_vals[i], i, p))
            env.model.save_model(p, env.iteration)
            if params_path is None:
                return
            p = params_path.format(name=name, metric=metric, val=metric_vals[i])
            print('[%d] saving params %s' % (env.iteration, p))
            params_copy = env.params.copy()
            params_copy.update({'num_iterations': int(improved_iterations[-1] + 1)})
            with open(p, 'w') as fp:
                json.dump(params_copy, fp, sort_keys=True, indent=1)
        else:
            print('[%d] metric %s:%s did not improve. Last improved on iteration %d' %
                  (env.iteration, name, metric, improved_iterations[-1]))

    callback.order = 99
    return callback


def print_feature_importance(print_every_iterations=10, importance_type='gain'):
    """LGBM callback."""
    def callback(env):
        if env.iteration % print_every_iterations != 0:
            return
        names = env.model.feature_name()
        ivals = env.model.feature_importance(importance_type)
        print('[%d] feature importance' % env.iteration)
        p = len('[%d]' % env.iteration)
        for i in np.argsort(-1 * ivals):
            print('%s %-52s %.3lf' % (' ' * p, names[i], ivals[i]))

    callback.order = 99
    return callback

# Regular expressions and function for parsing musicians.
RE_MUSICIANS_SPLIT_PATTERN = re.compile(r'feat(.)\w*|\(|\)|\||\/')


def parse_musicians(mstr):
    mm = []
    if type(mstr) is str:
        mstr = mstr.lower().replace('\n', ' ')
        s = re.sub(RE_MUSICIANS_SPLIT_PATTERN, ',', mstr).split(',')
        for t in s:
            t = t.strip()
            if len(t) > 0:
                mm.append(t)
    return mm


def parse_genres(gstr):
    if type(gstr) is not str:
        return []
    return [int(x) for x in gstr.split('|')]


def parse_isrc_year(isrc):
    if type(isrc) is not str:
        return isrc
    y1 = int(isrc[5:7])
    y0 = 19 if y1 > 17 else 20
    return int('%d%02d' % (y0, y1))


def round_to_nearest(n, r):
    return int(round(n / r) * r)


def encoder(series, v2i=None):
    """Encode a series to the smallest possible numerical type"""
    v2i = v2i or {np.nan: -1}
    dtypes = [np.int8, np.int16, np.int32, np.int64]
    n = len(series.unique())
    for dtype in dtypes:
        if n < np.iinfo(dtype).max:
            break
    encoded = np.empty(len(series), dtype=dtype)
    for si, v in enumerate(series.values):
        i = v2i.get(v)
        if i is None:
            i = len(v2i)
            v2i[v] = i
        encoded[si] = i
    return encoded


class GBDTRec(object):

    def __init__(self, artifacts_dir, data_dir='data',
                 interaction_space_params=INTERACTION_SPACE_PARAMS_DEFAULT):
        self.artifacts_dir = artifacts_dir
        self.interaction_space_params = interaction_space_params
        self.data_dir = data_dir
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_base_features(self, feats):

        # Rename and encode user and song ids.
        feats['user_id_cat'] = encoder(feats['msno'])
        feats['song_id_cat'] = encoder(feats['song_id'])
        feats.drop('msno', axis=1, inplace=True)
        feats.drop('song_id', axis=1, inplace=True)

        #########
        # USERS #
        #########

        # Get unique users and transform them.
        cols_user = ['user_id_cat', 'bd', 'city', 'gender', 'registration_init_time']
        U = feats[cols_user].drop_duplicates('user_id_cat')
        self.logger.info('Transforming %d unique users' % len(U))

        # Clip ages and replace unreasonable values with np.nan.
        fix_age = lambda x: min(max(x, 10), 60) if 10 <= x <= 60 else np.nan
        U['user_age_con'] = U['bd'].apply(fix_age)
        U.drop('bd', axis=1, inplace=True)

        # Rename and encode cities.
        U['user_city_cat'] = encoder(U['city'])
        U.drop('city', axis=1, inplace=True)

        # Rename and encode gender.
        U['user_gender_cat'] = encoder(U['gender'])
        U.drop('gender', axis=1, inplace=True)

        # Extract the registration year.
        get_year = lambda t: int(str(t)[:4])
        U['user_regyear_con'] = U['registration_init_time'].apply(get_year)
        U['user_regyear_con'] = U['user_regyear_con'].astype(np.uint16)

        # Count user plays.
        user_id_counter = Counter(feats['user_id_cat'].values)
        U['user_plays_con'] = U['user_id_cat'].apply(user_id_counter.get)

        # Keep only the new columns.
        U = U[['user_id_cat', 'user_age_con', 'user_city_cat',
               'user_gender_cat', 'user_regyear_con', 'user_plays_con']]

        #########
        # SONGS #
        #########

        # Get unique songs and transform them.
        cols_song = ['song_id_cat', 'artist_name', 'composer', 'genre_ids', 'isrc',
                     'language', 'lyricist', 'song_length']
        S = feats[cols_song].drop_duplicates('song_id_cat')
        self.logger.info('Transforming %d unique songs' % len(S))

        # Replace unknown artists, lyricists, composers. '佚名' means missing names.
        # https://www.kaggle.com/c/kkbox-music-recommendation-challenge/discussion/43645
        S['artist_name'].replace('佚名', np.nan, inplace=True)
        S['lyricist'].replace('佚名', np.nan, inplace=True)
        S['composer'].replace('佚名', np.nan, inplace=True)

        # Replace languages. 3,10,24,59 all correspond to Chinese.
        # https://www.kaggle.com/c/kkbox-music-recommendation-challenge/discussion/43645
        S['language'].replace(10, 3, inplace=True)
        S['language'].replace(24, 3, inplace=True)
        S['language'].replace(59, 3, inplace=True)

        # Replace negative language value with positive.
        S['language'].replace(-1, 1, inplace=True)
        S['song_language_cat'] = encoder(S['language'])

        # Compute song hashes. Later these will replace the song IDs.
        self.logger.info('Hashing songs')
        song_hash_cols = list(set(cols_song) - {'song_id_cat'})
        get_hash = lambda row: md5(str(row.values.tolist()).encode()).hexdigest()
        S['song_hash_cat'] = encoder(S[song_hash_cols].apply(get_hash, axis=1))

        # Leave song length as is and rename.
        S['song_len_con'] = S['song_length']
        S.drop('song_length', axis=1, inplace=True)

        # Extract song country and year from ISRC.
        get_country = lambda x: x[:2] if type(x) is str else np.nan
        S['song_country_cat'] = S['isrc'].apply(get_country)
        S['song_year_con'] = S['isrc'].apply(parse_isrc_year)

        # Keep and rename the raw musicians.
        S.rename({
            'artist_name': 'song_artist_raw_cat',
            'composer': 'song_composer_raw_cat',
            'lyricist': 'song_lyricist_raw_cat',
            'genre_ids': 'song_genres_raw_cat'},
            axis='columns', inplace=True)

        # Parse and count musicians and genres. Pick the most common one for each row.
        self.logger.info('Parsing, counting musicians and genres')
        song_id_2_musicians = {}
        song_id_2_genres = {}
        song_id_counter = Counter(feats['song_id_cat'].values)
        musicians_counter = Counter()
        genres_counter = Counter()
        for i, row in S.iterrows():
            song_count = song_id_counter[row['song_id_cat']]
            mm = parse_musicians(row['song_artist_raw_cat'])
            mm += parse_musicians(row['song_composer_raw_cat'])
            mm += parse_musicians(row['song_lyricist_raw_cat'])
            mm = list(set(mm))
            song_id_2_musicians[row['song_id_cat']] = mm
            musicians_counter.update(mm * song_count)
            gg = parse_genres(row['song_genres_raw_cat'])
            song_id_2_genres[row['song_id_cat']] = gg
            genres_counter.update(gg * song_count)

        self.logger.info('Frequent musicians: %s' % str(musicians_counter.most_common()[:10]))
        self.logger.info('Frequent genres: %s' % str(genres_counter.most_common()[:10]))
        self.logger.info('Replacing musicians and genres')
        musicians, genres = [], []
        for k in S['song_id_cat'].values:
            mm = song_id_2_musicians[k]
            if len(mm) > 0:
                cc = [musicians_counter[m] for m in mm]
                musicians.append(mm[np.argmax(cc)])
            else:
                musicians.append(np.nan)
            gg = song_id_2_genres[k]
            if len(gg) > 0:
                cc = [genres_counter[g] for g in gg]
                genres.append(gg[np.argmax(cc)])
            else:
                genres.append(np.nan)

        S['song_musician_cat'] = musicians
        S['song_genre_cat'] = genres
        S['song_genre_cat'] = S['song_genre_cat']

        # Count song, musician, genre plays.
        S['song_id_plays_con'] = S['song_id_cat'].apply(song_id_counter.get)
        S['song_musician_plays_con'] = S['song_musician_cat'].apply(musicians_counter.get)
        S['song_genre_plays_con'] = S['song_genre_cat'].apply(genres_counter.get)

        # Keep only the new columns.
        S = S[['song_id_cat', 'song_hash_cat', 'song_len_con', 'song_country_cat',
               'song_year_con', 'song_musician_cat', 'song_genre_cat', 'song_language_cat',
               'song_artist_raw_cat', 'song_lyricist_raw_cat', 'song_composer_raw_cat',
               'song_genres_raw_cat', 'song_id_plays_con', 'song_musician_plays_con',
               'song_genre_plays_con']]

        ###########
        # CONTEXT #
        ###########

        # Clean up and encode context variables.
        feats['source_screen_name'].replace('Unknown', np.nan, inplace=True)
        feats['source_system_tab'].replace('null', np.nan, inplace=True)
        feats['ctxt_scr_cat'] = encoder(feats['source_screen_name'])
        feats['ctxt_tab_cat'] = encoder(feats['source_system_tab'])
        feats['ctxt_type_cat'] = encoder(feats['source_type'])
        feats.drop('source_screen_name', axis=1, inplace=True)
        feats.drop('source_system_tab', axis=1, inplace=True)
        feats.drop('source_type', axis=1, inplace=True)

        # Keep subset of columns.
        feats = feats[['user_id_cat', 'song_id_cat', 'ctxt_scr_cat', 'ctxt_tab_cat',
                       'ctxt_type_cat', 'target']]

        ###########
        # MERGING #
        ###########

        # Left join feats with the users and songs.
        feats = feats.merge(U, on='user_id_cat', how='left')
        feats = feats.merge(S, on='song_id_cat', how='left')

        # Replace the song id with song hash.
        feats['song_id_cat'] = feats['song_hash_cat']
        return feats.drop('song_hash_cat', axis=1)

    def _get_interaction_features(self, feats_base):

        # Count training and testing rows.
        nb_trn = len(feats_base) - sum(feats_base['target'].isnull())
        nb_tst = len(feats_base) - nb_trn

        # Make a copy of feats_base for manipulation.
        cols_user = [c for c in feats_base.columns if c.startswith('user_')]
        cols_song = [c for c in feats_base.columns if c.startswith('song_')]
        feats_intr = feats_base[cols_user + cols_song + ['target']].copy()

        self.logger.info('Discretizing continuous features')
        round_age = lambda x: x if np.isnan(x) else round_to_nearest(x, 3)
        feats_intr['user_age_cat'] = feats_intr['user_age_con'].apply(round_age)
        feats_intr.drop('user_age_con', axis=1, inplace=True)

        round_len = lambda x: x if np.isnan(x) else int(round(np.log(x)))
        feats_intr['song_len_cat'] = feats_intr['song_len_con'].apply(round_len)
        feats_intr.drop('song_len_con', axis=1, inplace=True)

        round_year = lambda x: x if np.isnan(x) else round_to_nearest(x, 3)
        feats_intr['user_regyear_cat'] = feats_intr['user_regyear_con'].apply(round_year)
        feats_intr.drop('user_regyear_con', axis=1, inplace=True)

        feats_intr['song_year_cat'] = feats_intr['song_year_con'].apply(round_year)
        feats_intr.drop('song_year_con', axis=1, inplace=True)

        # Replace missing values with a common token.
        MISSING = '_UNK_'
        feats_intr.fillna(MISSING, inplace=True)

        self.logger.info('Encoding unique feature values')
        feats_lookup, nb_feats = {}, 1
        for c in feats_intr.columns:
            feats_lookup[c] = {f: nb_feats + i for i, f in enumerate(feats_intr[c].unique())}
            feats_lookup[MISSING] = 0
            nb_feats += len(feats_lookup[c])

        self.logger.info('Found %d unique feature values' % nb_feats)

        # Compute product of all user and song columns.
        cols_user = [c for c in feats_intr.columns if c.startswith('user_')]
        cols_song = [c for c in feats_intr.columns if c.startswith('song_')]
        cols_prod = list(product(cols_user, cols_song))

        # Populate training and test index pairs and the training targets.
        X_trn = np.empty((nb_trn * len(cols_prod), 2), dtype=np.uint32)
        X_tst = np.empty((nb_tst * len(cols_prod), 2), dtype=np.uint32)
        y_trn = np.empty((nb_trn * len(cols_prod),), dtype=np.uint8)

        self.logger.info('Training interactions: %d (%.3lf GB)' % (X_trn.shape[0], X_trn.nbytes / 10e8))
        self.logger.info('Testing  interactions: %d (%.3lf GB)' % (X_tst.shape[0], X_tst.nbytes / 10e8))

        for pi, (c0, c1) in enumerate(cols_prod):
            self.logger.info('Populating interaction %d of %d: (%s, %s)' % (pi, len(cols_prod), c0, c1))

            # Indexing into the matrix being populated.
            i0_trn = pi * nb_trn
            i0_tst = pi * nb_tst

            # Apply prefixes to the first column and then translate to indexes.
            X_ = feats_intr[c0].apply(feats_lookup[c0].get)
            X_trn[i0_trn:i0_trn + nb_trn, 0] = X_[:nb_trn]
            X_tst[i0_tst:i0_tst + nb_tst, 0] = X_[nb_trn:]

            # Apply prefixes to the second column, then translate to indexes.
            X_ = feats_intr[c1].apply(feats_lookup[c1].get)
            X_trn[i0_trn:i0_trn + nb_trn, 1] = X_[:nb_trn]
            X_tst[i0_tst:i0_tst + nb_tst, 1] = X_[nb_trn:]

            # Copy over the targets.
            y_trn[i0_trn:i0_trn + nb_trn] = feats_intr['target'].iloc[:nb_trn].values

        # Initialize and fit the vector space model.
        self.interaction_space_params.update({'nb_vecs': nb_feats})
        model = InteractionSpaceModel(**self.interaction_space_params)
        model.fit(X_trn, y_trn)
        yp_trn = model.transform(X_trn).astype(np.float16)
        yp_tst = model.transform(X_tst).astype(np.float16)

        # Append the pairwise cosine similarities as columns in feats_base.
        for pi, (c0, c1) in enumerate(cols_prod):
            c = 'sim_%s_%s_con' % (c0, c1)
            self.logger.info('Populating column %s' % c)
            i0_trn = pi * nb_trn
            i0_tst = pi * nb_tst
            feats_base[c] = np.concatenate([
                yp_trn[i0_trn:i0_trn + nb_trn],
                yp_tst[i0_tst:i0_tst + nb_tst]
            ])

        return feats_base

    def get_features(self, which='train'):

        assert which in {'train', 'test'}
        feats_intr, feats_base = None, None
        path_feats_base = '%s/feats-base.csv' % self.artifacts_dir
        path_feats_intr = '%s/feats-interactions.csv' % self.artifacts_dir
        ready_feats_base = os.path.exists(path_feats_base)
        ready_feats_intr = os.path.exists(path_feats_intr)

        if not ready_feats_base:
            t0 = time()
            self.logger.info('Reading dataframes')
            feats_sng = pd.read_csv('%s/songs.csv' % self.data_dir)
            feats_sei = pd.read_csv('%s/song_extra_info.csv' % self.data_dir)
            feats_mmb = pd.read_csv('%s/members.csv' % self.data_dir)
            feats_trn = pd.read_csv('%s/train.csv' % self.data_dir)
            feats_tst = pd.read_csv('%s/test.csv' % self.data_dir)

            self.logger.info('Merging dataframes')
            feats_trn = feats_trn.merge(feats_mmb, on='msno', how='left')
            feats_tst = feats_tst.merge(feats_mmb, on='msno', how='left')
            feats_trn = feats_trn.merge(feats_sng, on='song_id', how='left')
            feats_tst = feats_tst.merge(feats_sng, on='song_id', how='left')
            feats_trn = feats_trn.merge(feats_sei, on='song_id', how='left')
            feats_tst = feats_tst.merge(feats_sei, on='song_id', how='left')

            self.logger.info('Combining feats_trn and feats_tst')
            feats = feats_trn.append(feats_tst, ignore_index=True)

            # Save some memory.
            del feats_sng, feats_sei, feats_mmb, feats_trn, feats_tst

            # Encode test and train rows at the same time.
            self.logger.info('Engineering base features')
            feats_base = self._get_base_features(feats)
            feats_base.to_csv(path_feats_base, index=False)
            self.logger.info('Completed in %d seconds' % (time() - t0))

        # if not ready_feats_intr:
        #     t0 = time()
        #     self.logger.info('Engineering interaction features')
        #     feats_base = pd.read_csv(path_feats_base)
        #     feats_intr = self._get_interaction_features(feats_base)
        #     feats_intr.to_csv(path_feats_intr, index=False)
        #     self.logger.info('Completed in %d seconds' % (time() - t0))

        # if feats_intr == None:
        #     feats_intr = pd.read_csv(path_feats_intr)

        # for c in feats_intr.columns:
        #     if c.endswith('_cat'):
        #         feats_intr[c] = feats_intr[c].astype('category')
        # nb_trn = len(feats_intr) - sum(feats_intr['target'].isnull())

        # if which == 'train':
        #     return feats_intr.iloc[:nb_trn]
        # elif which == 'test':
        #     return feats_intr.iloc[nb_trn:]

        if feats_base is None:
            feats_base = pd.read_csv(path_feats_base)
        for c in feats_base.columns:
            if c.endswith('_cat'):
                feats_base[c] = feats_base[c].astype('category')
        nb_trn = len(feats_base) - sum(feats_base['target'].isnull())
        X, y = feats_base.drop('target', axis=1), feats_base['target']

        if which == 'train':
            return X.iloc[:nb_trn], y.iloc[:nb_trn]
        elif which == 'test':
            return X.iloc[nb_trn:]

    def val(self, X, y, val_prop=0.2, gbdt_params=GBDT_PARAMS_DEFAULT):

        self.logger.info('Preparing datasets')
        X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=val_prop, shuffle=False)

        self.logger.info('Converting dataset to lgbm format')
        gbdt_trn = lgbm.Dataset(X_trn, y_trn)
        gbdt_val = lgbm.Dataset(X_val, y_val, reference=gbdt_trn)

        self.logger.info('Training')
        model_path = '%s/model-{name:s}-{metric:s}-{val:.2f}.txt' % self.artifacts_dir
        params_path = '%s/model-{name:s}-{metric:s}-{val:.2f}.json' % self.artifacts_dir
        gbdt_cb = [
            save_best_model(10, 'val', 'auc', 'max', model_path, params_path),
            print_feature_importance(10)
        ]
        gbdt = lgbm.train(
            gbdt_params, train_set=gbdt_trn, valid_sets=[gbdt_trn, gbdt_val],
            valid_names=['trn', 'val'], verbose_eval=10, callbacks=gbdt_cb)

    def fit(self, X, y, gbdt_params):

        if type(gbdt_params) is str:
            with open(gbdt_params) as fp:
                gbdt_params = json.load(fp)
        assert type(gbdt_params) is dict

        self.logger.info('Preparing dataset')
        gbdt_trn = lgbm.Dataset(X, y)

        self.logger.info('Training')
        self.logger.info('GBDT Params\n%s' % pformat(gbdt_params))
        model_path = '%s/model-{name:s}-{metric:s}-{val:.2f}.txt' % self.artifacts_dir
        gbdt_cb = [
            save_best_model(10, 'trn', 'auc', 'max', model_path),
            print_feature_importance(10)
        ]
        gbdt = lgbm.train(
            gbdt_params, train_set=gbdt_trn, valid_sets=[gbdt_trn],
            valid_names=['trn'], verbose_eval=10, callbacks=gbdt_cb)

    def predict(self, X, gbdt_model, gbdt_params):

        if type(gbdt_params) is str:
            with open(gbdt_params) as fp:
                gbdt_params = json.load(fp)
        assert type(gbdt_params) is dict
        self.logger.info('GBDT Params\n%s' % pformat(gbdt_params))

        assert type(gbdt_model) is str
        self.logger.info('Loading model from %s' % gbdt_model)
        gbdt = lgbm.Booster(model_file=gbdt_model, silent=False)

        self.logger.info('Making predictions')
        yp = gbdt.predict(X, num_iteration=gbdt.current_iteration())
        self.logger.info('Target mean = %.2lf' % yp.mean())
        df = pd.DataFrame({'id': np.arange(len(yp)), 'target': yp})
        submission_path = gbdt_model.replace('.txt', '-submission.csv')
        df.to_csv(submission_path, index=False)
        self.logger.info('Saved predictions %s' % submission_path)


class MaskedVecDot(Layer):

    def __init__(self, **kwargs):
        super(MaskedVecDot, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs):
        VI_0, VI_1, V_0, V_1 = inputs
        V_dot = K.sum(V_0 * V_1, axis=-1)
        return V_dot * K.clip(VI_0 * VI_1, 0, 1)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class InteractionSpaceModel(object):

    def __init__(self, nb_dims, nb_vecs, vecs_init_func, vecs_init_kwargs,
                 optimizer, optimizer_kwargs, batch_size, nb_epochs_max):
        self.nb_dims = nb_dims
        self.nb_vecs = nb_vecs
        self.vecs_init_func = vecs_init_func
        self.vecs_init_kwargs = vecs_init_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.batch_size = batch_size
        self.nb_epochs_max = nb_epochs_max
        self.net = self._get_net()

    def _get_net(self):
        # Define, initialize vector space.
        self.vecs_init_kwargs.update({'size': (self.nb_vecs, self.nb_dims)})
        Vw = self.vecs_init_func(**self.vecs_init_kwargs)
        Vw[0] *= 0
        V = Embedding(self.nb_vecs, self.nb_dims, name='vecs', weights=[Vw])

        # Inputs and vector dot product.
        VI_0, VI_1 = Input((1,)), Input((1,))
        V_0, V_1 = V(VI_0), V(VI_1)
        V_dot = MaskedVecDot()([VI_0, VI_1, V_0, V_1])

        # Apply sigmoid activation for pseudo-classification.
        clsf = Activation('sigmoid')(V_dot)

        # Model with two inputs, out output.
        return Model([VI_0, VI_1], clsf)

    def fit(self, X, y):
        opt = self.optimizer(**self.optimizer_kwargs)
        self.net.compile(opt, loss='binary_crossentropy', metrics=['accuracy'])
        self.net.fit([X[:, 0], X[:, 1]], y, batch_size=self.batch_size,
                     epochs=self.nb_epochs_max)

    def transform(self, X):
        return self.net.predict([X[:, 0], X[:, 1]], batch_size=self.batch_size, verbose=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description='')
    ap.add_argument('--fit', action='store_true', default=False)
    ap.add_argument('--val', action='store_true', default=False)
    ap.add_argument('--predict', action='store_true', default=False)
    ap.add_argument('--params', type=str, default=None)
    ap.add_argument('--model', type=str, default=None)
    args = vars(ap.parse_args())

    rec = GBDTRec(artifacts_dir='artifacts/gbdtrec')

    if args['fit']:
        rec.fit(*rec.get_features('train'),
                gbdt_params=args['params'] or GBDT_PARAMS_DEFAULT)

    if args['val']:
        rec.val(*rec.get_features('train'))

    if args['predict']:
        rec.predict(X=rec.get_features('test'),
                    gbdt_model=args['model'],
                    gbdt_params=args['params'] or GBDT_PARAMS_DEFAULT)
