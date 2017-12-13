from collections import Counter
from hashlib import md5
from more_itertools import flatten
from pprint import pformat
from scipy.sparse import csr_matrix, save_npz, load_npz
from time import time
from tqdm import tqdm
import argparse
import json
import lightgbm as lgbm
import logging
import numpy as np
import os
import pandas as pd
import pdb
import re

GBDT_PARAMS_DEFAULT = {

    # Defining the task.
    'application': 'binary',
    'objective': 'binary',
    'boosting': 'gbdt',
    'metric': 'auc',
    'train_metric': True,

    # How many learners to fit, and how long to continue without
    # improvement on the validation set.
    'num_iterations': 475,
    'early_stopping_rounds': 50,

    # TODO: explain these parameters.
    'learning_rate': 0.4,
    'max_bin': 255,

    # Constraints on the tree characteristics.
    # Generally larger values will fit better but may over-fit.
    'max_depth': 10,
    'num_leaves': 64,

    # Randomly select *bagging_fraction* of the data to fit a learner.
    # Perform bagging at every *bagging_freq* iterations.
    # Seed the random bagging with *bagging_seed*.
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'bagging_seed': 1,

    # Randomly select *feature_fraction* of the features to fit a learner.
    # Seed the random selection with *feature_fraction_seed*.
    'feature_fraction': 1.0,
    'feature_fraction_seed': 1,

}

# Regular expressions and function for parsing musicians.
RE_MUSICIANS_SPLIT_PATTERN = re.compile(r'feat(.)\w*|\(|\)|\||\/')


def save_best_model(check_every_iterations, name, metric, mode, model_path, params_path=None):
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


class GBDTRec(object):

    def __init__(self, artifacts_dir, data_dir='data'):
        self.artifacts_dir = artifacts_dir
        self.data_dir = data_dir
        self.logger = logging.getLogger(self.__class__.__name__)

    def _feature_transformations(self, CMB):

        # Rename user and song ids for consistency.
        # Others will be renamed later.
        CMB.rename({'msno': 'user_id_cat', 'song_id': 'song_id_cat'},
                   axis='columns', inplace=True)

        #########
        # USERS #
        #########

        # Get unique users and transform them.
        user_cols = ['user_id_cat', 'bd', 'city', 'gender', 'registration_init_time']
        U = CMB[user_cols].drop_duplicates('user_id_cat')
        self.logger.info('Transforming %d unique users' % len(U))

        # Clip the ages at a reasonable value and rename.
        U['bd'].clip(5, 70, inplace=True)
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
        S = CMB[song_cols].drop_duplicates('song_id_cat')
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

        # Compute song hashes. Later these will replace the song IDs.
        self.logger.info('Hashing songs')
        get_hash = lambda row: md5(str(row.values.tolist()).encode()).hexdigest()
        S['song_hash_cat'] = S[song_hash_cols].apply(get_hash, axis=1)

        # Leave song length as is and rename.
        S.rename({'song_length': 'song_len_con'}, axis='columns', inplace=True)

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
        song_id_counter = Counter(CMB['song_id_cat'].values)
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
               'song_year_con', 'song_musician_cat', 'song_genre_cat']]

        ###########
        # CONTEXT #
        ###########

        # Clean up context variables.
        CMB['source_screen_name'].replace('Unknown', np.nan, inplace=True)
        CMB['source_system_tab'].replace('null', np.nan, inplace=True)
        CMB.rename({'source_screen_name': 'ctxt_scr_cat',
                    'source_system_tab': 'ctxt_tab_cat',
                    'source_type': 'ctxt_type_cat'},
                   axis='columns', inplace=True)

        # Keep subset of columns.
        CMB = CMB[['user_id_cat', 'song_id_cat', 'ctxt_scr_cat', 'ctxt_tab_cat',
                   'ctxt_type_cat', 'target']]

        ###########
        # MERGING #
        ###########

        # Left join CMB with the users and songs.
        CMB = CMB.merge(U, on='user_id_cat', how='left')
        CMB = CMB.merge(S, on='song_id_cat', how='left')

        # Replace the song id with song hash.
        CMB['song_id_cat'] = CMB['song_hash_cat']
        CMB.drop('song_hash_cat', inplace=True, axis=1)
        return CMB

    def get_features(self, which='train'):

        assert which in {'train', 'test'}

        path_trn = '%s/data-trn.csv' % self.artifacts_dir
        path_tst = '%s/data-tst.csv' % self.artifacts_dir
        feats_ready = os.path.exists(path_trn) and os.path.exists(path_tst)
        TRN = TST = None

        if not feats_ready:
            t0 = time()
            self.logger.info('Reading dataframes')
            SNG = pd.read_csv('%s/songs.csv' % self.data_dir)
            SEI = pd.read_csv('%s/song_extra_info.csv' % self.data_dir)
            MMB = pd.read_csv('%s/members.csv' % self.data_dir)
            TRN = pd.read_csv('%s/train.csv' % self.data_dir)
            TST = pd.read_csv('%s/test.csv' % self.data_dir)

            self.logger.info('Merging dataframes')
            TRN = TRN.merge(MMB, on='msno', how='left')
            TST = TST.merge(MMB, on='msno', how='left')
            TRN = TRN.merge(SNG, on='song_id', how='left')
            TST = TST.merge(SNG, on='song_id', how='left')
            TRN = TRN.merge(SEI, on='song_id', how='left')
            TST = TST.merge(SEI, on='song_id', how='left')

            self.logger.info('Combining TRN and TST')
            CMB = TRN.append(TST, ignore_index=True)

            nb_trn = len(TRN)
            nb_tst = len(TST)

            # Save some memory.
            del SNG, SEI, MMB, TRN, TST

            # Encode test and train rows at the same time.
            self.logger.info('Transforming features')
            CMB = self._feature_transformations(CMB)
            TRN = CMB.iloc[:nb_trn]
            TST = CMB.iloc[-nb_tst:]
            TRN.to_csv(path_trn, index=False)
            TST.to_csv(path_tst, index=False)
            self.logger.info('Completed in %d seconds' % (time() - t0))

        if which == 'train':
            if TRN is None:
                TRN = pd.read_csv(path_trn)
            for c in TRN.columns:
                if c.endswith('_cat'):
                    TRN[c] = TRN[c].astype('category')
            return TRN

        elif which == 'test':
            if TST is None:
                TST = pd.read_csv(path_tst)
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
