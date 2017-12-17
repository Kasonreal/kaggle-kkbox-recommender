from collections import Counter
from hashlib import md5
from itertools import product
from more_itertools import flatten, chunked
from pprint import pformat
from scipy.sparse import csc_matrix
from shutil import copyfile
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from subprocess import call
from time import time
from tqdm import tqdm
import argparse
import json
import logging
import numpy as np
import os
import pandas as pd
import pdb

import xlearn as xl

XLFM_PARAMS_DEFAULT = {
    'task': 'binary',
    'lr': 0.02,
    'lambda': 0.0002,
    'epoch': 30,
    'k': 10,
    'metric': 'acc'
}


class XLFMRec(object):

    def __init__(self, artifacts_dir, data_dir='data'):
        self.artifacts_dir = artifacts_dir
        self.data_dir = data_dir
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_features(self, D, nb_trn, path_trn, path_tst):
        """
        Output format:
        y1 idx:value idx:value ...
        y2 idx:value idx:value ...
        """

        MISSING_NUM = np.nan
        MISSING_STR = '__MISSING__'

        D.rename({'msno': 'u_id_cat', 'song_id': 's_id_cat'}, axis='columns', inplace=True)

        # Transform each unique user.
        user_cols = ['u_id_cat', 'bd', 'city', 'gender', 'registration_init_time']
        U = D[user_cols].drop_duplicates('u_id_cat')
        self.logger.info('Transforming %d unique users' % len(U))

        # Clip ages and replace unreasonable values with misisng token.
        fix_age = lambda x: min(max(x, 5), 70) if 10 <= x <= 70 else MISSING_NUM
        U.rename({'bd': 'u_age_con'}, axis='columns', inplace=True)
        U['u_age_con'] = U['u_age_con'].apply(fix_age)

        # Rename and encode cities. No missing values.
        U.rename({'city': 'u_city_cat'}, axis='columns', inplace=True)

        # Rename and encode gender.
        U.rename({'gender': 'u_gender_cat'}, axis='columns', inplace=True)
        U['u_gender_cat'].fillna(MISSING_STR, inplace=True)
        U['u_gender_cat'] = U['u_gender_cat']

        # Extract the registration year.
        get_year = lambda t: int(str(t)[:4])
        U['u_regyear_con'] = U['registration_init_time'].apply(get_year)
        U['u_regyear_con'] = U['u_regyear_con'].astype(np.uint16)

        # Count user plays.
        user_id_counter = Counter(D['u_id_cat'].values)
        U['u_plays_cat'] = U['u_id_cat'].apply(user_id_counter.get)
        U['u_plays_cat'] = np.log(U['u_plays_cat']).round()

        # Get the relevant columns.
        user_cols = [c for c in U.columns if c.startswith('u_')]
        U = U[user_cols]

        # Transform each unique song.
        # song_cols = ['s_id_cat', 'artist_name', 'composer',
        #              'genre_ids', 'isrc', 'language', 'lyricist', 's_length']
        song_cols = ['s_id_cat', 'artist_name', 'composer', 'lyricist', 'genre_ids']
        S = D[song_cols].drop_duplicates('s_id_cat')
        self.logger.info('Transforming %d unique songs' % len(S))
        S.rename({'artist_name': 's_artist_cat', 'composer': 's_composer_cat',
                  'lyricist': 's_lyricist_cat', 'genre_ids': 's_genres_cat'
                  }, axis='columns', inplace=True)

        # Transform context features.
        D['source_screen_name'].replace('Unknown', np.nan, inplace=True)
        D['source_system_tab'].replace('null', np.nan, inplace=True)
        D.rename({'source_system_tab': 'c_1_cat', 'source_screen_name': 'c_2_cat',
                  'source_type': 'c_3_cat'}, axis='columns', inplace=True)

        # Merge transformed user and song features back into original dataframe.
        D = D[['target', 'u_id_cat', 's_id_cat', 'c_1_cat', 'c_2_cat', 'c_3_cat']]
        D = D.merge(U, on='u_id_cat', how='left')
        D = D.merge(S, on='s_id_cat', how='left')

        # Dummy for test rows.
        D['target'].fillna(0, inplace=True)

        # Book-keeping for fast feature encoding.
        encoders = {c: {} for c in D.columns}
        missing_set = {MISSING_NUM, MISSING_STR}
        col_2_idx = {c: i for i, c in enumerate(sorted(D.columns[1:]))}
        ctxt_cols = [c for c in D if c.startswith('c_')]
        user_cols = [c for c in D if c.startswith('u_')]
        song_cols = [c for c in D if c.startswith('s_')]
        uid_2_str = {}
        sid_2_str = {}

        # Encode a row as its string of features.
        def encode_row(row, cols, skip_cold):
            s = ''
            for c in cols:
                if row[c] in missing_set:
                    continue
                v = encoders[c].get(row[c])
                if v == None:
                    if skip_cold:
                        continue
                    v = len(encoders[c])
                    encoders[c][row[c]] = v
                s = '%s %d:%d:1' % (s, col_2_idx[c], v)
            return s

        # Write many lines to file in batches.
        # Builds the *_2_str dicts to avoid re-computing fetaure strings.
        def write_lines(fp, rowiter, size, batch_size, skip_cold):
            batch_sum = 0
            batch = []
            for i in tqdm(range(size)):
                _, row = next(rowiter)
                user_str = uid_2_str.get(row['u_id_cat'])
                if user_str is None:
                    user_str = encode_row(row, user_cols, skip_cold)
                    uid_2_str[row['u_id_cat']] = user_str
                song_str = sid_2_str.get(row['s_id_cat'])
                if song_str is None:
                    song_str = encode_row(row, song_cols, skip_cold)
                    sid_2_str[row['s_id_cat']] = song_str
                ctxt_str = encode_row(row, ctxt_cols, skip_cold)
                batch.append('%d %s %s %s\n' % (row['target'], ctxt_str, song_str, user_str))
                if len(batch) % batch_size == 0:
                    fp.write(''.join(batch))
                    batch_sum += len(batch)
                    batch = []
            fp.write(''.join(batch))

        with open(path_trn, 'w') as fp:
            write_lines(fp, D.iloc[:nb_trn].iterrows(), nb_trn, 100000, False)
        copyfile(path_trn, '%s.bak' % path_trn)

        with open(path_tst, 'w') as fp:
            write_lines(fp, D.iloc[nb_trn:].iterrows(), len(D) - nb_trn, 100000, True)
        copyfile(path_tst, '%s.bak' % path_tst)

    def get_features(self, which='train'):

        assert which in {'train', 'test'}
        path_trn = '%s/feats-trn.txt' % self.artifacts_dir
        path_tst = '%s/feats-tst.txt' % self.artifacts_dir

        if not (os.path.exists(path_trn) and os.path.exists(path_tst)):
            t0 = time()
            self.logger.info('Reading dataframes')
            df_sng = pd.read_csv('%s/songs.csv' % self.data_dir)
            df_sei = pd.read_csv('%s/song_extra_info.csv' % self.data_dir)
            df_mmb = pd.read_csv('%s/members.csv' % self.data_dir)
            df_trn = pd.read_csv('%s/train.csv' % self.data_dir)
            df_tst = pd.read_csv('%s/test.csv' % self.data_dir)

            self.logger.info('Merging dataframes')
            df_trn = df_trn.merge(df_mmb, on='msno', how='left')
            df_tst = df_tst.merge(df_mmb, on='msno', how='left')
            df_trn = df_trn.merge(df_sng, on='song_id', how='left')
            df_tst = df_tst.merge(df_sng, on='song_id', how='left')
            df_trn = df_trn.merge(df_sei, on='song_id', how='left')
            df_tst = df_tst.merge(df_sei, on='song_id', how='left')

            self.logger.info('Combining feats_trn and feats_tst')
            df_cmb = df_trn.append(df_tst, ignore_index=True)
            del df_sng, df_sei, df_mmb, df_trn, df_tst

            # Encode test and train rows at the same time.
            self.logger.info('Engineering features')
            nb_trn = len(df_cmb) - sum(df_cmb['target'].isnull())
            self._get_features(df_cmb, nb_trn, path_trn, path_tst)
            self.logger.info('Completed in %d seconds' % (time() - t0))

        if which == 'train':
            return path_trn
        else:
            return path_tst

    def _split_features(self, path, val_prop=0.2):
        nb_lines = 0
        with open(path) as fp:
            for l in fp:
                nb_lines += 1
        nb_val = int(nb_lines * val_prop)
        nb_trn = nb_lines - nb_val

        self.logger.info('Training samples: %d' % nb_trn)
        self.logger.info('Validation samples: %d' % nb_val)

        path_trn = path.replace('.txt', '-trn.txt')
        path_val = path.replace('.txt', '-val.txt')
        call(["head -n%d %s > %s" % (nb_trn, path, path_trn)], shell=True)
        call(["tail -n%d %s > %s" % (nb_val, path, path_val)], shell=True)
        return path_trn, path_val

    def val(self, path_trn, val_prop=0.2, xlfm_params=XLFM_PARAMS_DEFAULT):
        self.logger.info('Preparing datasets')
        path_trn, path_val = self._split_features(path_trn, val_prop)
        fm_model = xl.create_fm()
        fm_model.setTrain(path_trn)
        fm_model.setValidate(path_val)
        fm_model.fit(xlfm_params, '%s/model-ffm-best-val.out' % self.artifacts_dir)

    def fit(self, path_trn, xlfm_params=XLFM_PARAMS_DEFAULT):
        ffm_model = xl.create_ffm()
        ffm_model.setTrain(path_trn)
        ffm_model.fit(xlfm_params, '%s/model-ffm-best-trn.out' % self.artifacts_dir)

    def predict(self, path_tst, model_path, xlfm_params=XLFM_PARAMS_DEFAULT):
        submission_path = '%s/submission-%d.csv' % (self.artifacts_dir, int(time()))
        ffm_model = xl.create_ffm()
        ffm_model.setTest(path_tst)
        ffm_model.setSigmoid()
        ffm_model.predict(model_path, submission_path)
        with open(submission_path) as fp:
            yp = [float(l.strip()) for l in fp]
        df = pd.DataFrame({'id': list(range(len(yp))), 'target': yp})
        df['target'] = df['target'].astype(np.float32)
        self.logger.info('yp mean %.3lf' % df['target'].mean())
        self.logger.info('%d rows' % len(df['target']))
        df.to_csv(submission_path, index=False)
        self.logger.info('Saved %s' % submission_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description='')
    ap.add_argument('--fit', action='store_true', default=False)
    ap.add_argument('--val', action='store_true', default=False)
    ap.add_argument('--predict', action='store_true', default=False)
    ap.add_argument('--model', type=str, default=None)
    args = vars(ap.parse_args())

    rec = XLFMRec(artifacts_dir='artifacts/xlfmrec')

    if args['fit']:
        rec.fit(rec.get_features('train'), xlfm_params=XLFM_PARAMS_DEFAULT)

    if args['val']:
        rec.val(rec.get_features('train'), xlfm_params=XLFM_PARAMS_DEFAULT)

    if args['predict']:
        rec.predict(rec.get_features('test'), model_path=args['model'])
