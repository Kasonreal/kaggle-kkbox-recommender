from collections import Counter
from hashlib import md5
from more_itertools import flatten
from scipy.sparse import csr_matrix, save_npz, load_npz
from time import time
from tqdm import tqdm
import argparse
import json
import lightgbm as lgb
import logging
import numpy as np
import os
import pandas as pd
import pdb
import re


LGB_HYPERPARAMS = {
    'objective': 'binary',
    'boosting': 'gbdt',
    'learning_rate': 0.2,
    'verbose': 0,
    'num_leaves': 100,
    'bagging_fraction': 0.95,
    'bagging_freq': 1,
    'bagging_seed': 1,
    'feature_fraction': 0.9,
    'feature_fraction_seed': 1,
    'max_bin': 256,
    'num_rounds': 100,
    'metric': 'auc'
}

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

    def val(self, data, val_prop=0.2, lgb_hyperparams=LGB_HYPERPARAMS):

        # Split data.
        nb_trn = int(len(data) * (1 - val_prop))
        data_trn, data_val = data.iloc[:nb_trn], data.iloc[nb_trn:]
        X_cols = [c for c in data.columns if c != 'target']
        X_trn, y_trn = data_trn[X_cols], data_trn['target']
        X_val, y_val = data_val[X_cols], data_val['target']

        lgb_trn = lgb.Dataset(X_trn, y_trn)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_trn)

        params = {
            'objective': 'binary',
            'boosting': 'gbdt',
            'learning_rate': 0.3,
            'num_leaves': 108,
            'bagging_fraction': 0.95,
            'bagging_freq': 1,
            'bagging_seed': 1,
            'feature_fraction': 0.9,
            'feature_fraction_seed': 1,
            'max_bin': 256,
            'max_depth': 10,
            'num_boost_round': 1500,
            'metric': 'auc'
        }

        model = lgb.train(params, train_set=lgb_trn,  valid_sets=lgb_val, verbose_eval=10)
        pdb.set_trace()

        # # LGB model with SKlearn API.
        # lgb_hyperparams = {
        #     'objective': 'binary',
        #     'boosting_type': 'gbdt',
        #     'num_leaves': 2,
        #     'max_depth': -1,
        #     'learning_rate': 0.1,
        #     'n_estimators': 1,
        #     'silent': False
        # }
        # gbdt = lgb.LGBMModel(**lgb_hyperparams)
        # gbdt.fit(X_trn, y_trn, eval_set=(X_val, y_val), eval_metric='auc', verbose=True)

    def fit(self, X=None, y=None, names=None, categ=None, lgb_hyperparams=LGB_HYPERPARAMS):
        best_model_path = '%s/model-%d' % (self.artifacts_dir, int(time()))
        lgbm = LGBMModel(**lgbm_hyperparams)
        return

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description='')
    ap.add_argument('--fit', action='store_true', default=False)
    ap.add_argument('--val', action='store_true', default=False)
    args = vars(ap.parse_args())

    rec = GBDTRec(artifacts_dir='artifacts/gbdtrec')

    if args['fit']:
        rec.fit(data=rec.get_features('train'))

    if args['val']:
        rec.val(data=rec.get_features('train'))
