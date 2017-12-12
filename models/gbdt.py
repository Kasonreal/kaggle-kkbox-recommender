from hashlib import md5
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


LGBM_HYPERPARAMS = {
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
    # 'max_bin': 256,
    # 'num_rounds': 100,
    'metric': 'auc'
}


class GBDTRec(object):

    def __init__(self, artifacts_dir, data_dir='data'):
        self.artifacts_dir = artifacts_dir
        self.data_dir = data_dir
        self.logger = logging.getLogger(self.__class__.__name__)

    def _features_df_to_csr(self, CMB):

        feat2col = {}     # Mapping feature -> csr column.
        user2csr = {}     # Mapping user id -> (data, cols).
        song2csr = {}     # Mapping song id -> (data, cols).

        def getval(dict_, key):
            val = dict_.get(key)
            if val is None:
                val = len(dict_)
                dict_[key] = val
            return dict_, val

        # Regular expressions and function for parsing musicians.
        RE_MUSICIANS_SPLIT_PATTERN = re.compile(r'feat(.)\w*|\(|\)|\||\/')

        def parse_musicians(mstr):
            r = []
            if type(mstr) is str:
                mstr = mstr.lower().replace('\n', ' ')
                s = re.sub(RE_MUSICIANS_SPLIT_PATTERN, ',', mstr).split(',')
                for t in s:
                    t = t.strip()
                    if len(t) > 0:
                        r.append(t)
            return r

        # Get unique users and encode their features in CSR format.
        user_cols = ['msno', 'bd', 'city', 'gender', 'registration_init_time', 'registered_via']
        U = CMB[user_cols].drop_duplicates('msno')
        self.logger.info('Encoding %d unique users' % len(U))
        for i, row in tqdm(U.iterrows()):
            csr_data, csr_cols = [], []

            # User ID is a categorical feature.
            feat = 'cat-user-msno-%s' % row['msno']
            feat2col, col = getval(feat2col, feat)
            csr_data.append(1)
            csr_cols.append(col)

            # User age is a continuous feature.
            if 0 < row['bd'] < 100:
                feat = 'con-user-age'
                feat2col, col = getval(feat2col, feat)
                csr_data.append(row['bd'])
                csr_cols.append(col)

            # User city is a categorical feature.
            feat = 'cat-user-city-%d' % row['city']
            feat2col, col = getval(feat2col, feat)
            csr_data.append(1)
            csr_cols.append(col)

            # User gender is a categorical feature.
            if type(row['gender']) == str:
                feat = 'cat-user-gender-%s' % row['gender']
                feat2col, col = getval(feat2col, feat)
                csr_data.append(1)
                csr_cols.append(col)

            user2csr[row['msno']] = (csr_data, csr_cols)

        # Get unique songs and encode their features in CSR format.
        song_cols = ['song_id', 'artist_name', 'composer', 'genre_ids', 'isrc',
                     'language', 'lyricist', 'name', 'song_length']
        song_hash_cols = list(set(song_cols) - {'song_id'})
        S = CMB[song_cols].drop_duplicates('song_id')
        self.logger.info('Encoding %d unique songs' % len(S))

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

        for i, row in tqdm(S.iterrows()):

            csr_data, csr_cols = [], []

            # Replace the song id with a song hash because some of the songs
            # have identical features but different ids.
            song_hash = md5(str(row[song_hash_cols].values).encode()).hexdigest()
            feat = 'cat-song-hash-%s' % song_hash
            feat2col, col = getval(feat2col, feat)
            csr_data.append(1)
            csr_cols.append(col)

            # Song artists, lyricists, composers are categorical features.
            mm = []
            for c in ['artist_name', 'lyricist', 'composer']:
                if type(row[c]) != str:
                    continue
                mm += parse_musicians(row[c])
            for m in set(mm):
                feat = 'cat-song-musician-%s' % m.replace(' ', '-')
                feat2col, col = getval(feat2col, feat)
                csr_data.append(1)
                csr_cols.append(col)

            # Genres are categorical features.
            if type(row['genre_ids']) == str:
                for g in row['genre_ids'].split('|'):
                    feat = 'cat-song-genre-%d' % int(g)
                    feat2col, col = getval(feat2col, feat)
                    csr_data.append(1)
                    csr_cols.append(col)

            # Language is a categorical feature.
            if not np.isnan(row['language']):
                feat = 'cat-song-language-%d' % int(row['language'])
                feat2col, col = getval(feat2col, feat)
                csr_data.append(1)
                csr_cols.append(col)

            # Song length is a continuous feature.
            # User age is a continuous feature.
            if not np.isnan(row['song_length']):
                feat = 'con-song-length'
                feat2col, col = getval(feat2col, feat)
                csr_data.append(int(row['song_length'] / 1000))
                csr_cols.append(col)

            # Song country is a categorical feature.
            # Song year is a continuous feature.
            if type(row['isrc']) is str:
                feat = 'cat-song-country-%s' % row['isrc'][:2]
                feat2col, col = getval(feat2col, feat)
                csr_data.append(1)
                csr_cols.append(col)

                year_suffix = row['isrc'][5:7]
                year_prefix = '20' if int(year_suffix) < 18 else '19'
                year = int('%s%s' % (year_prefix, year_suffix))
                feat = 'con-song-year'
                feat2col, col = getval(feat2col, feat)
                csr_data.append(year)
                csr_cols.append(col)

            song2csr[row['song_id']] = (csr_data, csr_cols)

        # Clean up context variables.
        CMB['source_screen_name'].replace('Unknown', np.nan, inplace=True)
        CMB['source_system_tab'].replace('null', np.nan, inplace=True)

        # Iterate over all rows and encode their features in CSR format.
        # Use the previously computed unique user and song features.
        cmb_cols = ['msno', 'song_id', 'source_type', 'source_system_tab', 'source_screen_name']
        csr_data, csr_rows, csr_cols = [], [], []
        self.logger.info('Encoding %d samples' % len(CMB))
        for i, row in tqdm(CMB[cmb_cols].iterrows()):
            len_pre = len(csr_data)

            # Add context features.
            if type(row['source_type']) == str:
                feat = 'cat-ctxt-source-type-%s' % row['source_type'].replace(' ', '-')
                feat2col, col = getval(feat2col, feat)
                csr_data.append(1)
                csr_cols.append(col)

            if type(row['source_system_tab']) == str:
                feat = 'cat-ctxt-source-tab-%s' % row['source_system_tab'].replace(' ', '-')
                feat2col, col = getval(feat2col, feat)
                csr_data.append(1)
                csr_cols.append(col)

            if type(row['source_screen_name']) == str:
                feat = 'cat-ctxt-source-screen-%s' % row['source_screen_name'].replace(' ', '-')
                feat2col, col = getval(feat2col, feat)
                csr_data.append(1)
                csr_cols.append(col)

            # Add pre-computed user and song features.
            user_data, user_cols = user2csr[row['msno']]
            song_data, song_cols = song2csr[row['song_id']]
            csr_data += user_data
            csr_data += song_data
            csr_cols += user_cols
            csr_cols += song_cols
            csr_rows += [i] * (len(csr_data) - len_pre)

        X_csr = csr_matrix((csr_data, (csr_rows, csr_cols)),
                           shape=(len(CMB), len(feat2col)),
                           dtype=np.float32)

        col2feat = {c: f for f, c in feat2col.items()}
        names = [col2feat[c] for c in range(len(col2feat))]
        categ = [f for f in feat2col.keys() if f.startswith('cat-')]
        return X_csr, names, categ

    def get_features(self, which='train'):

        assert which in {'train', 'test'}

        X_trn_path = '%s/data-X-trn.npz' % self.artifacts_dir
        X_tst_path = '%s/data-X-tst.npz' % self.artifacts_dir
        names_path = '%s/data-feats-names.json' % self.artifacts_dir
        categ_path = '%s/data-feats-categ.json' % self.artifacts_dir
        feats_paths = [X_trn_path, X_tst_path, names_path, categ_path]
        feats_ready = sum(map(os.path.exists, feats_paths)) == len(feats_paths)

        if not feats_ready:
            t0 = time()
            self.logger.info('Reading dataframes')
            SNG = pd.read_csv('%s/songs.csv' % self.data_dir)
            SEI = pd.read_csv('%s/song_extra_info.csv' % self.data_dir)
            MMB = pd.read_csv('%s/members.csv' % self.data_dir)
            TRN = pd.read_csv('%s/train.csv' % self.data_dir, nrows=1000)
            TST = pd.read_csv('%s/test.csv' % self.data_dir, nrows=1000)

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
            self.logger.info('Encoding features as CSR matrix')
            X_csr, names, categ = self._features_df_to_csr(CMB)
            self.logger.info(X_csr.__repr__())
            self.logger.info('%d unique features' % len(names))
            self.logger.info('%d categorical features' % len(categ))

            # Then separate them when serializing.
            save_npz(X_trn_path, X_csr[:nb_trn])
            save_npz(X_tst_path, X_csr[-nb_tst:])
            with open(names_path, 'w') as fp:
                json.dump(names, fp, indent=1)
            with open(categ_path, 'w') as fp:
                json.dump(categ, fp, indent=1)
            self.logger.info('Preprocessed, saved features in %d seconds' % (time() - t0))

        with open(names_path) as fp:
            names = json.load(fp)

        with open(categ_path) as fp:
            categ = json.load(fp)

        if which == 'train':
            X = load_npz(X_trn_path)
            df = pd.read_csv('%s/train.csv' % self.data_dir, usecols=['target'], nrows=X.shape[0])
            y = df['target'].values
            return X, y, names, categ

        elif which == 'test':
            X = load_npz(X_tst_path)
            return X, names, categ

    def val(self, X, y, names, categ, val_prop=0.2, lgbm_hyperparams=LGBM_HYPERPARAMS):
        nb_trn = int(X.shape[0] * (1 - val_prop))
        ds_arg = {'feature_name': names, 'categorical_feature': categ}
        ds_trn = lgb.Dataset(X[:nb_trn], y[:nb_trn], **ds_arg).construct()
        ds_val = lgb.Dataset(X[nb_trn:], y[nb_trn:], reference=ds_trn, **ds_arg).construct()
        pdb.set_trace()

    def fit(self, X=None, y=None, names=None, categ=None, lgbm_hyperparams=LGBM_HYPERPARAMS):
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
        rec.fit(*rec.get_features('train'))

    if args['val']:
        rec.val(*rec.get_features('train'))
