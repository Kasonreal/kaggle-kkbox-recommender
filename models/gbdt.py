from collections import Counter
from hashlib import md5
from itertools import product
from more_itertools import flatten
from pprint import pformat
from scipy.sparse import csr_matrix, save_npz, load_npz
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

# from keras.optimizers import Adagrad

GBDT_PARAMS_DEFAULT = {

    # Defining the task.
    'objective': 'binary',
    'boosting': 'gbdt',
    'metric': 'auc',
    'train_metric': True,

    # How many learners to fit, and how long to continue without
    # improvement on the validation set.
    'num_iterations': 1000,
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
    'vec_size': 50,
    'vecs_init_func': np.random.normal,
    'vecs_init_kwargs': {'loc': 0, 'scale': 0.05},
    # 'optimizer': Adagrad,
    'optimizer_kwargs': {'lr': 0.01},
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
            print('%s %-20s %.3lf' % (' ' * p, names[i], ivals[i]))

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
    return round(n / r) * r


class GBDTRec(object):

    def __init__(self, artifacts_dir, data_dir='data',
                 interaction_space_params=INTERACTION_SPACE_PARAMS_DEFAULT):
        self.artifacts_dir = artifacts_dir
        self.interaction_space_params = interaction_space_params
        self.data_dir = data_dir
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_base_features(self, feats_cmb):

        # Rename user and song ids for consistency.
        # Others will be renamed later.
        feats_cmb.rename({'msno': 'user_id_cat', 'song_id': 'song_id_cat'},
                         axis='columns', inplace=True)

        #########
        # USERS #
        #########

        # Get unique users and transform them.
        user_cols = ['user_id_cat', 'bd', 'city', 'gender', 'registration_init_time']
        U = feats_cmb[user_cols].drop_duplicates('user_id_cat')
        self.logger.info('Transforming %d unique users' % len(U))

        # Clip the ages at a reasonable value and rename. Replace unreasonable
        # values with NaNs.
        fix_age = lambda x: min(max(x, 10), 60) if 10 <= x <= 60 else np.nan
        U['bd'] = U['bd'].apply(fix_age)
        U.rename({'bd': 'user_age_con'}, axis='columns', inplace=True)

        # Keep city as-is and rename.
        U.rename({'city': 'user_city_cat'}, axis='columns', inplace=True)

        # Gender has missing values which are nan. Just rename it.
        U.rename({'gender': 'user_gender_cat'}, axis='columns', inplace=True)

        # Extract the registration year.
        get_year = lambda t: int(str(t)[:4])
        U['user_regyear_con'] = U['registration_init_time'].apply(get_year)

        # Keep only the new columns.
        U = U[['user_id_cat', 'user_age_con', 'user_city_cat',
               'user_gender_cat', 'user_regyear_con']]

        #########
        # SONGS #
        #########

        # Get unique songs and transform them.
        song_cols = ['song_id_cat', 'artist_name', 'composer', 'genre_ids', 'isrc',
                     'language', 'lyricist', 'song_length']
        song_hash_cols = list(set(song_cols) - {'song_id_cat'})
        S = feats_cmb[song_cols].drop_duplicates('song_id_cat')
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

        # Compute song hashes. Later these will replace the song IDs.
        self.logger.info('Hashing songs')
        get_hash = lambda row: md5(str(row.values.tolist()).encode()).hexdigest()
        S['song_hash_cat'] = S[song_hash_cols].apply(get_hash, axis=1)

        # Leave song length as is and rename.
        S.rename({'song_length': 'song_len_con'}, axis='columns', inplace=True)
        pdb.set_trace()

        # Extract song country and year from ISRC.
        get_country = lambda x: x[:2] if type(x) is str else np.nan
        S['song_country_cat'] = S['isrc'].apply(get_country)
        S['song_year_con'] = S['isrc'].apply(parse_isrc_year)

        # Leave song language as is and rename.
        S.rename({'language': 'song_language_cat'}, axis='columns', inplace=True)

        # Parse and count musicians and genres. Pick the most common one for each row.
        self.logger.info('Parsing, counting musicians and genres')
        song_id_2_musicians = {}
        song_id_2_genres = {}
        song_id_counter = Counter(feats_cmb['song_id_cat'].values)
        musicians_counter = Counter()
        genres_counter = Counter()
        for i, row in S.iterrows():
            song_count = song_id_counter[row['song_id_cat']]
            mm = parse_musicians(row['artist_name'])
            mm += parse_musicians(row['lyricist'])
            mm += parse_musicians(row['composer'])
            mm = list(set(mm))
            song_id_2_musicians[row['song_id_cat']] = mm
            musicians_counter.update(mm * song_count)
            gg = parse_genres(row['genre_ids'])
            song_id_2_genres[row['song_id_cat']] = gg
            genres_counter.update(gg * song_count)

        self.logger.info('Frequent musicians: %s' % str(musicians_counter.most_common()[:5]))
        self.logger.info('Frequent genres: %s' % str(genres_counter.most_common()[:5]))
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

        # Keep only the new columns.
        S = S[['song_id_cat', 'song_hash_cat', 'song_len_con', 'song_country_cat',
               'song_year_con', 'song_musician_cat', 'song_genre_cat', 'song_language_cat']]

        ###########
        # CONTEXT #
        ###########

        # Clean up context variables.
        feats_cmb['source_screen_name'].replace('Unknown', np.nan, inplace=True)
        feats_cmb['source_system_tab'].replace('null', np.nan, inplace=True)
        feats_cmb.rename({'source_screen_name': 'ctxt_scr_cat',
                          'source_system_tab': 'ctxt_tab_cat',
                          'source_type': 'ctxt_type_cat'},
                         axis='columns', inplace=True)

        # Keep subset of columns.
        feats_cmb = feats_cmb[['user_id_cat', 'song_id_cat', 'ctxt_scr_cat', 'ctxt_tab_cat',
                               'ctxt_type_cat', 'target']]

        ###########
        # MERGING #
        ###########

        # Left join feats_cmb with the users and songs.
        feats_cmb = feats_cmb.merge(U, on='user_id_cat', how='left')
        feats_cmb = feats_cmb.merge(S, on='song_id_cat', how='left')

        # Replace the song id with song hash.
        feats_cmb['song_id_cat'] = feats_cmb['song_hash_cat']
        feats_cmb.drop('song_hash_cat', inplace=True, axis=1)
        return feats_cmb

    def _get_interaction_features(self, feats_base):

        self.logger.info('Discretizing continuous features')
        feats_intr = feats_base.copy()
        feats_intr = feats_intr[['user_id_cat', 'song_id_cat', 'song_musician_cat', 'song_genre_cat']]

        # round_age = lambda x: x if np.isnan(x) else round_to_nearest(x, 3)
        # feats_intr['user_age_cat'] = feats_intr['user_age_con'].apply(round_age).astype('category')
        # feats_intr.drop('user_age_con', axis=1, inplace=True)

        # round_year = lambda x: x if np.isnan(x) else round_to_nearest(x, 3)
        # feats_intr['user_regyear_cat'] = feats_intr['user_regyear_con'].apply(round_year).astype('category')
        # feats_intr.drop('user_regyear_con', axis=1, inplace=True)

        # feats_intr['song_len_cat'] = np.log(feats_intr['song_len_con']).round().astype('category')
        # feats_intr.drop('song_len_con', axis=1, inplace=True)

        # Encode all possible features into a term<->index vocabulary.
        # nan (missing) is encoded as index 0.
        feat_terms_all = set()
        for c in feats_intr.columns:
            set_prefix = lambda v: '%s::%s' % (c, str(v))
            feat_terms_new = map(set_prefix, feats_intr[c].unique().tolist())
            feat_terms_all = feat_terms_all.union(feat_terms_new)

        self.logger.info('Found %d unique feature terms' % len(feat_terms_all))

        pdb.set_trace()

        # Get the interaction product of user and song columns.
        cols_user = [c for c in feats_intr.columns if c.startswith('user_')]
        cols_song = [c for c in feats_intr.columns if c.startswith('song_')]
        cols_prod = list(product(cols_user, cols_song))

        pdb.set_trace()

        # Expand the rows into index pairs with a target.

        # Initialize and fit the vector space model.

        # Store the pairwise similarities as features.

        pdb.set_trace()

    def get_features(self, which='train'):

        assert which in {'train', 'test'}
        TRN = TST = None

        # FIXME: should be just feats-base.csv..
        path_feats_base = '%s/feats-base-trn.csv' % self.artifacts_dir
        path_feats_intr = '%s/feats-interactions.csv' % self.artifacts_dir
        path_model_intr = '%s/model-interaction-space.hdf5' % self.artifacts_dir

        ready_feats_base = os.path.exists(path_feats_base)
        ready_feats_intr = os.path.exists(path_feats_intr) and os.path.exists(path_model_intr)

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
            feats_cmb = feats_trn.append(feats_tst, ignore_index=True)

            # Save some memory.
            del feats_sng, feats_sei, feats_mmb, feats_trn, feats_tst

            # Encode test and train rows at the same time.
            self.logger.info('Engineering base features')
            feats_base = self._get_base_features(feats_cmb)
            feats_base.to_csv(path_feats_base, index=False)
            self.logger.info('Completed in %d seconds' % (time() - t0))

        if not ready_feats_intr:
            self.logger.info('Engineering interaction features')
            feats_base = pd.read_csv(path_feats_base, nrows=500000)
            feats_intr = self._get_interaction_features(feats_base)

            # ispace = InteractionSpace(**self.interaction_space_params)

        # TODO: compute interaction features and add them to the base features.

        if which == 'train':
            if TRN is None:
                TRN = pd.read_csv(path_base_trn, nrows=100000)
            for c in TRN.columns:
                if c.endswith('_cat'):
                    TRN[c] = TRN[c].astype('category')
            return TRN

        elif which == 'test':
            if TST is None:
                TST = pd.read_csv(path_base_tst)
            for c in TST.columns:
                if c.endswith('_cat'):
                    TST[c] = TST[c].astype('category')
            return TST

    def val(self, data, val_prop=0.2, gbdt_params=GBDT_PARAMS_DEFAULT):

        self.logger.info('Preparing datasets')
        nb_trn = int(len(data) * (1 - val_prop))
        data_trn, data_val = data.iloc[:nb_trn], data.iloc[nb_trn:]
        X_cols = [c for c in data.columns if c != 'target']
        X_trn, y_trn = data_trn[X_cols], data_trn['target']
        X_val, y_val = data_val[X_cols], data_val['target']

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

    def fit(self, data, gbdt_params):

        if type(gbdt_params) is str:
            with open(gbdt_params) as fp:
                gbdt_params = json.load(fp)
        assert type(gbdt_params) is dict

        self.logger.info('Preparing dataset')
        X_cols = [c for c in data.columns if c != 'target']
        gbdt_trn = lgbm.Dataset(data[X_cols], data['target'])

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

    def predict(self, data, gbdt_model, gbdt_params):

        if type(gbdt_params) is str:
            with open(gbdt_params) as fp:
                gbdt_params = json.load(fp)
        assert type(gbdt_params) is dict
        self.logger.info('GBDT Params\n%s' % pformat(gbdt_params))

        assert type(gbdt_model) is str
        self.logger.info('Loading model from %s' % gbdt_model)
        gbdt = lgbm.Booster(model_file=gbdt_model, silent=False)

        self.logger.info('Preparing dataset')
        X_cols = [c for c in data.columns if c not in {'id', 'target'}]

        self.logger.info('Making predictions')
        yp = gbdt.predict(data[X_cols], num_iteration=gbdt.current_iteration())
        self.logger.info('Target mean = %.2lf' % yp.mean())
        df = pd.DataFrame({'id': np.arange(len(yp)), 'target': yp})
        submission_path = gbdt_model.replace('.txt', '-submission.csv')
        df.to_csv(submission_path, index=False)
        self.logger.info('Saved predictions %s' % submission_path)


class InteractionSpace(object):

    def __init__(self, df1, df2, model_path):

        return

    def fit(self):

        return

    def eval(self):

        return

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
        rec.fit(data=rec.get_features('train'),
                gbdt_params=args['params'] or GBDT_PARAMS_DEFAULT)

    if args['val']:
        rec.val(data=rec.get_features('train'))

    if args['predict']:
        rec.predict(data=rec.get_features('test'), gbdt_model=args['model'],
                    gbdt_params=args['params'] or GBDT_PARAMS_DEFAULT)
